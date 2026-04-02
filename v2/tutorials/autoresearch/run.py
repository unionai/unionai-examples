# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "flyte>=2.0.0b22",
#     "PyGithub>=2.5.0",
#     "matplotlib>=3.7.0",
# ]
# ///

"""
AutoResearch Agent - Runs the autoresearch workflow using Claude Code CLI in a GPU environment.

This agent:
1. Starts a GPU-enabled container
2. Installs Claude Code CLI
3. Clones the autoresearch repository
4. Points Claude Code at program.md as the prompt and lets it run
5. Commits the result (CSV + code changes in train/) and creates a PR
"""

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from github import Auth, Github

import flyte
import flyte.report
from _image import image as autoresearch_image

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL = "parnianzargham@gmail.com"
AUTORESEARCH_REPO_URL = "https://github.com/unionai-oss/autoresearch.git"
AUTORESEARCH_REPO_FULL_NAME = "unionai-oss/autoresearch"


autoresearch_env = flyte.TaskEnvironment(
    name="autoresearch-agent",
    resources=flyte.Resources(
        cpu=8,
        memory="32Gi",
        gpu="T4:1",
        disk="100Gi",
    ),
    secrets=[
        flyte.Secret(key="github_token", as_env_var="GITHUB_TOKEN"),
        flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    ],
    image=autoresearch_image,
)


@dataclass
class AutoResearchResult:
    """Result of the autoresearch run."""

    pr_url: str
    pr_number: int
    branch_name: str
    files_changed: list[str]
    success: bool
    error_message: Optional[str] = None


def clone_repository(repo_url: str, work_dir: Path, github_token: str) -> Path:
    """Clone the autoresearch repository with authentication."""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = work_dir / repo_name

    # Inject token into HTTPS URL for authentication
    authenticated_url = repo_url.replace(
        "https://", f"https://{GITHUB_USERNAME}:{github_token}@"
    )

    if repo_path.exists():
        subprocess.run(["git", "pull"], cwd=repo_path, check=True)
    else:
        subprocess.run(["git", "clone", authenticated_url, str(repo_path)], check=True)

    return repo_path


@autoresearch_env.task(report=True)
async def run_autoresearch() -> AutoResearchResult:
    """
    Run the autoresearch workflow end-to-end.

    Steps:
    - Clone https://github.com/unionai-oss/autoresearch
    - Configure git identity
    - Create a new branch
    - Run Claude Code CLI with program.md as the prompt
    - Commit results (CSV + train/ changes)
    - Push and open a PR against the autoresearch repo
    """
    github_token = os.environ["GITHUB_TOKEN"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

    # --- Install Node.js + Claude Code at runtime (keeps image small and submission fast) ---
    import tarfile
    import urllib.request as _urllib

    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "git"], check=False)

    node_url = "https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.gz"
    node_tar = Path("/tmp/node.tar.gz")
    print(f"Downloading Node.js from {node_url}...", flush=True)
    _urllib.urlretrieve(node_url, node_tar)
    size_mb = node_tar.stat().st_size / 1024 / 1024
    print(f"Downloaded {size_mb:.1f} MB to {node_tar}", flush=True)
    if size_mb < 1:
        raise RuntimeError(f"Node.js download appears empty/corrupt ({size_mb:.2f} MB) — network may be restricted")
    node_dir = Path("/tmp/node")
    node_dir.mkdir(exist_ok=True)
    print("Extracting Node.js...", flush=True)
    with tarfile.open(node_tar, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.split("/", 1)[-1]]
        for m in members:
            m.name = m.name.split("/", 1)[-1]
        tar.extractall(str(node_dir), members=[m for m in members if m.name])

    # Add node/npm to PATH for this process and all subprocesses
    node_bin = str(node_dir / "bin")
    os.environ["PATH"] = node_bin + ":" + os.environ.get("PATH", "")
    print(f"Node version: {subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()}", flush=True)

    npm_prefix = "/tmp/npm-global"
    Path(npm_prefix).mkdir(exist_ok=True)
    subprocess.run(["npm", "install", "-g", "--prefix", npm_prefix, "@anthropic-ai/claude-code"], check=True)
    os.environ["PATH"] = str(Path(npm_prefix) / "bin") + ":" + os.environ["PATH"]
    print("Node.js + Claude Code installed.", flush=True)

    # --- Clone repo ---
    work_dir = Path("/tmp/autoresearch_workspace")
    work_dir.mkdir(exist_ok=True, parents=True)
    repo_path = clone_repository(AUTORESEARCH_REPO_URL, work_dir, github_token)

    # --- Git identity ---
    subprocess.run(
        ["git", "config", "--global", "user.email", GITHUB_EMAIL], check=True
    )
    subprocess.run(
        ["git", "config", "--global", "user.name", GITHUB_USERNAME], check=True
    )

    # --- Create branch ---
    import time as _time
    branch_name = f"autoresearch/claude-run-{int(_time.time())}"
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "checkout", branch_name],
            cwd=repo_path,
            check=True,
        )

    # --- Read program.md to use as the Claude Code prompt ---
    program_md = repo_path / "program.md"
    if not program_md.exists():
        raise FileNotFoundError(
            f"program.md not found in {repo_path}. "
            "Make sure the autoresearch repo has a program.md at its root."
        )

    program_md_content = program_md.read_text()
    print(f"Loaded prompt from program.md ({len(program_md_content)} chars)")

    # Install repo dependencies before handing off to Claude
    subprocess.run(["pip", "install", "matplotlib"], check=True)
    for pip_cmd in [
        ["pip", "install", "-e", "."],
        ["pip", "install", "-r", "requirements.txt"],
    ]:
        req_file = repo_path / pip_cmd[-1] if pip_cmd[-1].startswith("req") else None
        if req_file is None or req_file.exists():
            dep_result = subprocess.run(
                pip_cmd, cwd=repo_path, capture_output=True, text=True
            )
            print(f"{' '.join(pip_cmd)}:\n{dep_result.stdout}", flush=True)
            if dep_result.returncode != 0:
                print(f"(non-fatal) {dep_result.stderr}", flush=True)

    # Wrap the program.md content with explicit instructions to write outputs to disk
    prompt = f"""You are running inside an automated GPU pipeline. You MUST write all outputs to disk as actual files.

Here are your instructions from program.md:

{program_md_content}

LOGGING INSTRUCTIONS (follow exactly):
- Before you start any training, print this exact line: [AUTORESEARCH] Training started
- Before training, print what change you are testing: [AUTORESEARCH] Change: <one line description of the code change being tested>
- When training finishes, print this exact line: [AUTORESEARCH] Training finished
- After training, print the key metric value: [AUTORESEARCH] Metric: <metric name>=<value>
- When writing results to CSV, print this exact line: [AUTORESEARCH] Writing results to CSV

IMPORTANT: After completing the above instructions, make sure you have:
1. Written the final results to a CSV file in this repository (e.g. results/results.csv or similar)
2. Saved all code changes you made to the train/ directory (or wherever the training code lives)
3. All files must be written to the current working directory so they appear in git status
If any command fails, debug and fix it rather than stopping. Do not just print results — write them to files on disk."""

    # --- Pre-flight: verify claude is installed and API key is reachable ---
    version_check = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True
    )
    print(f"claude version: {version_check.stdout.strip()} | stderr: {version_check.stderr.strip()}", flush=True)
    if version_check.returncode != 0:
        raise RuntimeError(f"claude CLI not found or broken: {version_check.stderr}")

    # --- Disable Claude Code sandbox ---
    # In Kubernetes/Flyte pods, Claude Code's sandbox tries to spin up a nested container
    # which fails silently and causes file writes to go to an ephemeral space instead of
    # the real working directory. Disabling it makes writes land in the actual filesystem.
    claude_config_dir = Path("/root/.claude")
    claude_config_dir.mkdir(parents=True, exist_ok=True)
    settings = claude_config_dir / "settings.json"
    import json as _json
    existing = _json.loads(settings.read_text()) if settings.exists() else {}
    existing["sandbox"] = False
    settings.write_text(_json.dumps(existing, indent=2))
    print(f"Wrote Claude Code settings: {settings.read_text()}", flush=True)

    # --- Run Claude Code CLI ---
    # Matches swe_agent.py exactly: prompt as positional arg, CI=true enables non-interactive mode
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "--max-turns", "100",
        "--model", "claude-haiku-4-5-20251001",
        prompt,
    ]

    print(f"Running: {shlex.join(cmd[:3])} <prompt>", flush=True)

    claude_env = {
        **os.environ,
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "CLAUDE_SKIP_PERMISSIONS": "true",
        "CI": "true",  # Enables non-interactive mode (no TTY required)
    }

    # Stream output line by line so logs appear in real time instead of buffering until done
    proc = subprocess.Popen(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout stream
        text=True,
        env=claude_env,
    )

    stdout_lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        stdout_lines.append(line)

    proc.wait()
    full_output = "\n".join(stdout_lines)
    print(f"Claude Code exit code: {proc.returncode}", flush=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Claude Code CLI exited with code {proc.returncode}\n"
            f"output: {full_output[-2000:]}"
        )

    # --- Collect changed files ---
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    print(f"Git status:\n{git_status.stdout}", flush=True)

    files_changed = []
    for line in git_status.stdout.strip().splitlines():
        if line:
            # git status --porcelain: first two chars are XY status flags
            file_path = line[3:].strip()
            files_changed.append(file_path)

    # Also list all files in repo dir for debugging
    all_files = subprocess.run(
        ["find", ".", "-type", "f", "-not", "-path", "./.git/*"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    print(f"All files in repo:\n{all_files.stdout}", flush=True)

    if not files_changed:
        raise RuntimeError(
            "Claude Code ran successfully but produced no file changes.\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    # --- Commit ---
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "add", "-f", "results.tsv"], cwd=repo_path, check=False)
    subprocess.run(["git", "add", "-f", "results/"], cwd=repo_path, check=False)
    commit_message = (
        "feat: autoresearch run via Claude Code\n\n"
        "Added research results (CSV) and updated train/ code changes.\n"
        "Generated by the autoresearch Flyte agent."
    )
    subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_path,
        check=True,
    )

    # --- Push ---
    print(f"GitHub token present: {bool(github_token)}, length: {len(github_token) if github_token else 0}", flush=True)
    authenticated_url = AUTORESEARCH_REPO_URL.replace(
        "https://", f"https://{GITHUB_USERNAME}:{github_token}@"
    )
    subprocess.run(
        ["git", "remote", "set-url", "origin", authenticated_url],
        cwd=repo_path,
        check=True,
    )
    push_result = subprocess.run(
        ["git", "push", "-u", "origin", branch_name, "--force"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    print(f"Push stdout: {push_result.stdout}", flush=True)
    print(f"Push stderr: {push_result.stderr}", flush=True)
    if push_result.returncode != 0:
        raise RuntimeError(f"git push failed (exit {push_result.returncode}):\n{push_result.stderr}")

    # --- Create PR via PyGithub ---
    auth = Auth.Token(github_token)
    gh = Github(auth=auth)
    repo = gh.get_repo(AUTORESEARCH_REPO_FULL_NAME)

    csv_files = [f for f in files_changed if f.endswith(".csv")]
    train_files = [f for f in files_changed if "train" in f]

    pr_body = f"""## AutoResearch Run

This PR was automatically generated by the autoresearch Flyte agent using Claude Code CLI.

### What changed
- **Result CSV files**: {', '.join(f'`{f}`' for f in csv_files) or 'none detected'}
- **Train code changes**: {', '.join(f'`{f}`' for f in train_files) or 'none detected'}

### All changed files
{chr(10).join(f'- `{f}`' for f in files_changed)}

---
🤖 Generated by [autoresearch Flyte agent](https://github.com/unionai-oss/autoresearch)
"""

    existing_prs = list(repo.get_pulls(state="open", head=f"unionai-oss:{branch_name}"))
    if existing_prs:
        pr = existing_prs[0]
        print(f"PR already exists: {pr.html_url}", flush=True)
    else:
        pr = repo.create_pull(
            title="feat: autoresearch results + train changes",
            body=pr_body,
            head=branch_name,
            base="master",
        )
        print(f"PR created: {pr.html_url}", flush=True)

    # --- Generate Val BPB Over Time plot from results.tsv ---
    plot_path = repo_path / "val_bpb_over_time.png"
    results_tsv = repo_path / "results.tsv"
    if results_tsv.exists():
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kept_indices, kept_vals = [], []
        all_indices, all_vals = [], []

        with open(results_tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, row in enumerate(reader):
                try:
                    val = float(row["val_bpb"])
                except (KeyError, ValueError):
                    continue
                all_indices.append(i)
                all_vals.append(val)
                if row.get("status", "").strip().upper() == "KEEP":
                    kept_indices.append(i)
                    kept_vals.append(val)

        if all_vals:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(all_indices, all_vals, color="lightgray", s=20, label="All experiments", zorder=1)
            if kept_vals:
                ax.scatter(kept_indices, kept_vals, color="green", s=40, label="Kept", zorder=2)
                # Running minimum line
                running_min, running_min_idx = [], []
                current_min = float("inf")
                for idx, val in zip(kept_indices, kept_vals):
                    if val < current_min:
                        current_min = val
                    running_min_idx.append(idx)
                    running_min.append(current_min)
                ax.plot(running_min_idx, running_min, color="green", linewidth=2, label="Best so far", zorder=3)
            ax.set_xlabel("Experiment #")
            ax.set_ylabel("Val BPB")
            ax.set_title("Val BPB Over Time")
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(plot_path), dpi=150)
            plt.close(fig)
            print(f"Saved plot to {plot_path}", flush=True)

            # Upload plot to PR as a comment with base64 inline image
            import base64
            img_b64 = base64.b64encode(plot_path.read_bytes()).decode()
            pr_comment = (
                "## Val BPB Over Time\n\n"
                f"![Val BPB Over Time](data:image/png;base64,{img_b64})"
            )
            pr.create_issue_comment(pr_comment)
            print("Posted plot as PR comment.", flush=True)

            # Force-add plot to git and amend commit
            subprocess.run(["git", "add", "-f", str(plot_path)], cwd=repo_path, check=False)
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=repo_path, check=False,
            )
            subprocess.run(
                ["git", "push", "-u", "origin", branch_name, "--force"],
                cwd=repo_path, check=False,
            )

            # Show plot in Flyte UI via report
            await flyte.report.replace.aio(
                f"<h2>Val BPB Over Time</h2>"
                f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/>'
                f'<p><a href="{pr.html_url}">View PR</a></p>'
            )
            await flyte.report.flush.aio()
        else:
            print("results.tsv found but no valid val_bpb rows — skipping plot.", flush=True)
    else:
        print("results.tsv not found — skipping plot.", flush=True)

    return AutoResearchResult(
        pr_url=pr.html_url,
        pr_number=pr.number,
        branch_name=branch_name,
        files_changed=files_changed,
        success=True,
    )


if __name__ == "__main__":
    import time

    flyte.init_from_config()

    run = flyte.with_runcontext(mode="remote").run(run_autoresearch)

    print(f"AutoResearch run started: {run.url}")
    print("Waiting for completion...")

    while True:
        try:
            run.wait()
            break
        except Exception as e:
            print(f"Connection dropped ({e}), reconnecting in 30s...")
            time.sleep(30)

    print(f"Done! See run at: {run.url}")
