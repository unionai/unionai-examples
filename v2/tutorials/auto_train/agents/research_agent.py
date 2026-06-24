"""
ResearchAgent — launches Claude Code CLI to run the autoresearch loop.

All training iteration, metric parsing, progress tracking, git commits, and
keep/revert decisions are delegated to Claude Code CLI. program.md (written by
ArchitectureAgent) contains the full instructions for the loop.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL    = "parnianzargham@gmail.com"


# ---------------------------------------------------------------------------
# Runtime install helpers
# ---------------------------------------------------------------------------

_RESEARCHER = "researcher"
_RESEARCHER_HOME = Path(f"/home/{_RESEARCHER}")


def _install_node_and_claude() -> None:
    """
    Download Node.js, install Claude Code CLI, and create a non-root user.
    Claude Code CLI refuses --dangerously-skip-permissions when running as root,
    so we create a dedicated non-root user and run claude as that user.
    """
    subprocess.run(["apt-get", "update", "-y"], check=False, capture_output=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=False, capture_output=True)

    # Create non-root user (idempotent — ignore error if already exists)
    subprocess.run(["useradd", "-m", "-s", "/bin/bash", _RESEARCHER], check=False)
    print(f"Non-root user '{_RESEARCHER}' ready.", flush=True)

    node_url = "https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.gz"
    node_tar = Path("/tmp/node.tar.gz")
    print("Downloading Node.js...", flush=True)
    urllib.request.urlretrieve(node_url, node_tar)
    size_mb = node_tar.stat().st_size / 1024 / 1024
    print(f"  {size_mb:.1f} MB downloaded", flush=True)
    if size_mb < 1:
        raise RuntimeError(f"Node.js download appears corrupt ({size_mb:.2f} MB) — network may be restricted")

    node_dir = Path("/tmp/node")
    node_dir.mkdir(exist_ok=True)
    with tarfile.open(node_tar, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.split("/", 1)[-1]]
        for m in members:
            m.name = m.name.split("/", 1)[-1]
        tar.extractall(str(node_dir), members=[m for m in members if m.name])

    node_bin = str(node_dir / "bin")
    os.environ["PATH"] = node_bin + ":" + os.environ.get("PATH", "")
    ver = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"  Node {ver}", flush=True)

    npm_prefix = "/tmp/npm-global"
    Path(npm_prefix).mkdir(exist_ok=True)
    subprocess.run(
        ["npm", "install", "-g", "--prefix", npm_prefix, "@anthropic-ai/claude-code"],
        check=True, capture_output=True,
    )
    os.environ["PATH"] = str(Path(npm_prefix) / "bin") + ":" + os.environ["PATH"]
    print("Claude Code CLI installed.", flush=True)


def _disable_claude_sandbox(home: Path) -> None:
    """
    Disable Claude Code's sandbox for the given user home directory.
    In Kubernetes/Flyte pods, the sandbox tries to spin up a nested container
    which fails and causes file writes to land in an ephemeral space.
    """
    claude_config_dir = home / ".claude"
    claude_config_dir.mkdir(parents=True, exist_ok=True)
    settings = claude_config_dir / "settings.json"
    existing = json.loads(settings.read_text()) if settings.exists() else {}
    existing["sandbox"] = False
    settings.write_text(json.dumps(existing, indent=2))
    # Ensure the non-root user owns this config
    subprocess.run(["chown", "-R", f"{_RESEARCHER}:{_RESEARCHER}", str(claude_config_dir)], check=False)


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str, github_token: str):
        self.github_repo  = github_repo
        self.github_token = github_token

    def run(
        self,
        branch_name: str,
        experiment_folder: str,
        local_data_path: str,
        max_experiments: int = 20,
    ) -> str:
        """
        Install Claude CLI, clone the branch, run Claude with program.md as prompt.
        Returns the PR URL.
        """
        import flyte

        # 1. Install Node.js + Claude Code CLI + create non-root user
        _install_node_and_claude()
        _disable_claude_sandbox(_RESEARCHER_HOME)

        # 2. Clone the branch ArchitectureAgent already pushed
        clone_dir = self._clone(branch_name)
        exp_dir   = clone_dir / experiment_folder

        # 3. Make data readable by all + transfer repo ownership to non-root user
        #    Data dir was created as root; chmod so researcher can read it.
        subprocess.run(["chmod", "-R", "a+rX", local_data_path], check=False)
        subprocess.run(
            ["chown", "-R", f"{_RESEARCHER}:{_RESEARCHER}", str(clone_dir)],
            check=True,
        )

        # 4. Read program.md — embed it directly in the prompt, same as autoresearch run.py
        program_text = (exp_dir / "program.md").read_text()

        # 5. Build prompt matching autoresearch run.py style: minimal wrapper + program.md
        max_turns = max_experiments * 8
        prompt = f"""You are running inside an automated GPU pipeline on Union cloud. You MUST write all outputs to disk as actual files.

DATA_PATH environment variable is already set to: {local_data_path}

Here are your instructions:

{program_text}

LOGGING INSTRUCTIONS (follow exactly):
- Before each training run print: [AUTOTRAIN] Starting experiment <N>
- After parsing the metric print: [AUTOTRAIN] Result: <metric>=<value>
- After writing to progress.csv print: [AUTOTRAIN] CSV updated

IMPORTANT:
- Working directory is {exp_dir} — all files (train.py, best_train.py, progress.csv) are there
- DATA_PATH is already set in your environment; do not change it
- If any command fails, debug and fix it — do not stop the loop
- You MUST append a row to progress.csv after every experiment, even if the run crashed
- git remote is already authenticated; do not change the remote URL
"""

        # 7. Verify claude CLI is available
        ver_check = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if ver_check.returncode != 0:
            raise RuntimeError(f"claude CLI not found: {ver_check.stderr}")
        print(f"claude {ver_check.stdout.strip()}", flush=True)

        # 8. Run Claude Code CLI as non-root user
        #    --dangerously-skip-permissions is rejected when running as root,
        #    so we exec as the 'researcher' user created in _install_node_and_claude.
        #    max_turns: autoresearch uses 100; give more headroom for larger experiment budgets.
        max_turns = max(100, max_experiments * 15)
        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--max-turns", str(max_turns),
            "--model", self.MODEL,
            prompt,
        ]
        print(f"Starting Claude Code as '{_RESEARCHER}' (max_turns={max_turns}, max_experiments={max_experiments})...", flush=True)

        claude_env = {
            **os.environ,
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "DATA_PATH": local_data_path,
            "HOME": str(_RESEARCHER_HOME),
            "CI": "true",
            "CLAUDE_SKIP_PERMISSIONS": "true",
        }

        proc = subprocess.Popen(
            cmd,
            cwd=str(exp_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=claude_env,
            user=_RESEARCHER,
        )
        stdout_lines: list[str] = []
        exp_count = 0
        for line in proc.stdout:
            line = line.rstrip("\n")
            # Highlight [AUTOTRAIN] markers so experiment progress is visible in Union logs
            if "[AUTOTRAIN]" in line:
                if "Starting experiment" in line:
                    exp_count += 1
                    print(f"\n{'='*60}", flush=True)
                print(f">>> {line}", flush=True)
            else:
                print(line, flush=True)
            stdout_lines.append(line)
        proc.wait()

        full_output = "\n".join(stdout_lines)
        if proc.returncode != 0:
            if "max turns" in full_output.lower() or "reached max" in full_output.lower():
                print(
                    f"\nClaude reached max turns ({max_turns}) after ~{exp_count} experiments "
                    "— continuing with whatever was logged.",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"Claude Code CLI exited {proc.returncode}\n{full_output[-2000:]}"
                )

        # 9. Post-run diagnostics — log what Claude actually committed
        git_log = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=clone_dir, capture_output=True, text=True,
        )
        print(f"\nGit log after Claude session:\n{git_log.stdout}", flush=True)

        progress_csv = exp_dir / "progress.csv"
        print(f"\nprogress.csv contents:\n{progress_csv.read_text()}", flush=True)
        metric_name, higher = _parse_metric_direction(exp_dir / "program.md")
        plot_html = _build_progress_plot(progress_csv, metric_name, higher)
        best_value = _read_best_value(progress_csv, metric_name, higher)

        # 10. Create PR
        pr_url = self._create_pr(branch_name, exp_dir, metric_name, best_value)

        flyte.report.log(
            f"<h2>AutoTrain Progress — {branch_name}</h2>"
            f"{plot_html}"
            f'<p><a href="{pr_url}">View PR on GitHub</a></p>',
            do_flush=True,
        )

        return (
            f"AutoTrain complete\n"
            f"  Branch : {branch_name}\n"
            f"  Best {metric_name}: {best_value}\n"
            f"  PR     : {pr_url}"
        )

    # ------------------------------------------------------------------
    # Git
    # ------------------------------------------------------------------

    def _clone(self, branch_name: str) -> Path:
        clone_dir = Path(f"/tmp/automl_research_{branch_name}")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        repo_url = f"https://{self.github_token}@github.com/{self.github_repo}.git"
        subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
        subprocess.run(["git", "checkout", branch_name], cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.email", GITHUB_EMAIL], cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.name",  GITHUB_USERNAME], cwd=clone_dir, check=True)
        return clone_dir

    # ------------------------------------------------------------------
    # PR
    # ------------------------------------------------------------------

    def _create_pr(
        self,
        branch_name: str,
        exp_dir: Path,
        metric_name: str,
        best_value: str,
    ) -> str:
        try:
            from github import Auth, Github
            auth = Auth.Token(self.github_token)
            gh   = Github(auth=auth)
            repo = gh.get_repo(self.github_repo)

            default_branch = repo.default_branch
            existing = list(repo.get_pulls(
                state="open",
                head=f"{self.github_repo.split('/')[0]}:{branch_name}",
            ))
            if existing:
                print(f"PR already exists: {existing[0].html_url}", flush=True)
                return existing[0].html_url

            pr = repo.create_pull(
                title=f"AutoTrain: {branch_name} | best {metric_name}={best_value}",
                body=f"""## AutoTrain Results

| | |
|---|---|
| **Branch** | `{branch_name}` |
| **Metric** | `{metric_name}` |
| **Best value** | `{best_value}` |

### Files
- `train.py` — best training script found
- `best_train.py` — explicit backup
- `progress.csv` — full experiment history
- `program.md` — instructions given to Claude

---
Generated by [AutoTrain](https://github.com/unionai/unionai-examples/tree/main/v2/tutorials/auto_train)
""",
                head=branch_name,
                base=default_branch,
            )
            print(f"PR created: {pr.html_url}", flush=True)
            return pr.html_url
        except Exception as e:
            print(f"PR creation failed (non-fatal): {e}", flush=True)
            return f"https://github.com/{self.github_repo}/tree/{branch_name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_metric_direction(program_md: Path) -> tuple[str, bool]:
    import re
    text = program_md.read_text()
    m = re.search(r"BEST_VAL_(\w+):", text)
    metric_name = m.group(1).lower() if m else "metric"
    hib = re.search(r"higher_is_better:\s*(true|false)", text, re.IGNORECASE)
    higher = (hib.group(1).lower() == "true") if hib else True
    return metric_name, higher


def _read_best_value(progress_csv: Path, metric_name: str, higher: bool) -> str:
    try:
        rows = list(csv.DictReader(open(progress_csv)))
        values = [float(r["metric_value"]) for r in rows if r.get("metric_value", "").strip()]
        if not values:
            return "n/a"
        best = max(values) if higher else min(values)
        return f"{best:.6f}"
    except Exception:
        return "n/a"


def _build_progress_plot(progress_csv: Path, metric_name: str, higher: bool) -> str:
    try:
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.read_csv(progress_csv)
        df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
        df["improved"]     = df["improved"].astype(str).str.lower() == "true"

        kept     = df[df["improved"]]
        not_kept = df[~df["improved"]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=not_kept["experiment_id"], y=not_kept["metric_value"],
            mode="markers", name="No improvement",
            marker=dict(color="#cccccc", size=8),
        ))
        fig.add_trace(go.Scatter(
            x=kept["experiment_id"], y=kept["metric_value"],
            mode="markers", name="Improved",
            marker=dict(color="#2ecc71", size=12, symbol="star"),
        ))
        running = df["metric_value"].cummax() if higher else df["metric_value"].cummin()
        fig.add_trace(go.Scatter(
            x=df["experiment_id"], y=running,
            mode="lines", name="Running best",
            line=dict(color="#27ae60", width=2, dash="dash"),
        ))
        fig.update_layout(
            title=f"{metric_name} across experiments",
            xaxis_title="Experiment", yaxis_title=metric_name, height=400,
        )
        return fig.to_html(include_plotlyjs=True, full_html=False)
    except Exception as e:
        return f"<p>Plot unavailable: {e}</p>"
