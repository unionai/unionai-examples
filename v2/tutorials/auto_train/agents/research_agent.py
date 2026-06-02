"""
ResearchAgent — autoresearch-style iterative training loop.

1. Clones the GitHub branch created by ArchitectureAgent
2. Each experiment:
   a. Runs train.py as a subprocess (DATA_PATH env var, output streamed live)
   b. Parses BEST_VAL_{METRIC}: {value} from stdout
   c. If improved → saves as best_train.py, appends improved=True row to progress.csv
   d. If not improved → restores best_train.py, appends improved=False row
   e. Sends program.md + best_train.py + progress.csv to Claude → writes new train.py
   f. Commits and pushes train.py + progress.csv to the branch
3. Creates a PR via PyGithub and renders a progress plot in the Union UI report panel
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import anthropic

from agents.data_agent import DataProfile

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL    = "parnianzargham@gmail.com"

HIGHER_IS_BETTER = {
    "macro_f1": True, "accuracy": True, "roc_auc": True, "mcc": True, "r2": True,
    "rmse": False, "mae": False,
}


class ResearchAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str, github_token: str):
        self.github_repo  = github_repo
        self.github_token = github_token
        self.branch_name: str | None = None
        self.clone_dir:   Path | None = None
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        profile: DataProfile,
        branch_name: str,
        experiment_folder: str,
        max_experiments: int = 20,
        time_budget_per_experiment_seconds: float = 300.0,
    ) -> str:
        import flyte

        self.branch_name = branch_name
        self.clone_dir   = self._clone()

        exp_dir       = self.clone_dir / experiment_folder
        train_py      = exp_dir / "train.py"
        best_train_py = exp_dir / "best_train.py"
        program_md    = exp_dir / "program.md"
        progress_csv  = exp_dir / "progress.csv"

        program_text = program_md.read_text()
        m = re.search(r"BEST_VAL_(\w+):", program_text)
        metric_name = m.group(1).lower() if m else "macro_f1"
        higher = HIGHER_IS_BETTER.get(metric_name, True)

        best_value = float("-inf") if higher else float("inf")
        shutil.copy(train_py, best_train_py)

        # Install packages listed in ## Dependencies before the loop starts
        _install_from_program_md(program_text)

        print(f"\nResearchAgent: branch={self.branch_name}  folder={experiment_folder}  "
              f"metric={metric_name}  higher_is_better={higher}  max_experiments={max_experiments}\n",
              flush=True)

        for exp_id in range(max_experiments):
            print(f"{'='*60}", flush=True)
            print(f"Experiment {exp_id + 1}/{max_experiments}", flush=True)

            # 1. Run train.py — stream output line by line so logs appear live in Union UI
            t0 = time.time()
            stdout, rc = self._run_script_streaming(
                train_py, profile.local_data_path, time_budget_per_experiment_seconds
            )
            duration = round(time.time() - t0, 1)

            # 2. Parse metric
            value = _parse_metric(stdout, metric_name)
            notes = ""
            if value is None:
                notes = f"parse failed rc={rc} | last output: {stdout[-200:].strip()}"
                print(f"  FAILED: {notes}", flush=True)
                shutil.copy(best_train_py, train_py)
                value = best_value
                improved = False
            else:
                improved = _is_better(value, best_value, higher)
                print(f"  {metric_name}={value:.6f}  improved={improved}  {duration}s", flush=True)

            # 3. Keep or revert
            if improved:
                best_value = value
                shutil.copy(train_py, best_train_py)
                print(f"  New best: {best_value:.6f}", flush=True)
            else:
                shutil.copy(best_train_py, train_py)

            # 4. Append to progress.csv
            _append_csv(progress_csv, exp_id, train_py.read_text(),
                        metric_name, value, improved, duration, notes)

            # 5. Ask Claude for the next train.py
            new_train = self._ask_claude(
                program_text, best_train_py.read_text(),
                progress_csv.read_text(), metric_name, higher,
            )
            train_py.write_text(new_train)

            # Install any new packages Claude annotated with # REQUIRES: pkg1, pkg2
            _install_from_requires_comment(new_train)

            # 6. Commit + push
            self._commit_push(
                f"exp {exp_id+1}: {metric_name}={value:.4f} "
                f"{'[IMPROVED]' if improved else '[no improvement]'}",
            )

        # 7. Create PR
        pr_url = self._create_pr(progress_csv, metric_name, best_value, max_experiments)

        # 8. Render progress plot in Union UI
        plot_html = _build_progress_plot(progress_csv, metric_name, higher)
        flyte.report.log(
            f"<h2>AutoTrain Progress — {self.branch_name}</h2>"
            f"{plot_html}"
            f'<p><a href="{pr_url}">View PR on GitHub</a></p>',
            do_flush=True,
        )

        summary = _build_summary(progress_csv, metric_name, best_value, max_experiments,
                                  self.branch_name, pr_url)
        print(f"\n{summary}", flush=True)
        return summary

    # ------------------------------------------------------------------
    # Git
    # ------------------------------------------------------------------

    def _clone(self) -> Path:
        """Clone the branch that ArchitectureAgent already pushed to."""
        clone_dir = Path(f"/tmp/automl_research_{self.branch_name}")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        repo_url = f"https://{self.github_token}@github.com/{self.github_repo}.git"
        subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
        subprocess.run(["git", "checkout", self.branch_name], cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.email", GITHUB_EMAIL],    cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.name",  GITHUB_USERNAME], cwd=clone_dir, check=True)
        return clone_dir

    def _commit_push(self, message: str) -> None:
        subprocess.run(["git", "add", "."], cwd=self.clone_dir)
        subprocess.run(["git", "commit", "-m", message], cwd=self.clone_dir)
        subprocess.run(["git", "push", "origin", self.branch_name], cwd=self.clone_dir)

    # ------------------------------------------------------------------
    # Run train.py — streaming output
    # ------------------------------------------------------------------

    def _run_script_streaming(
        self, script: Path, data_path: str, budget: float
    ) -> tuple[str, int]:
        """Run train.py and stream each output line live. Returns (full_stdout, returncode)."""
        env      = {**os.environ, "DATA_PATH": data_path}
        lines:   list[str] = []
        deadline = time.time() + budget + 30

        proc = subprocess.Popen(
            ["python", str(script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr so everything streams together
            text=True,
        )

        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                print(f"  [train] {line}", flush=True)
                lines.append(line)
                if time.time() > deadline:
                    proc.kill()
                    lines.append("[TIMEOUT]")
                    break
        finally:
            proc.wait()

        return "\n".join(lines), proc.returncode

    # ------------------------------------------------------------------
    # Claude call — propose next train.py
    # ------------------------------------------------------------------

    def _ask_claude(
        self,
        program_md: str,
        best_train_py: str,
        progress_csv: str,
        metric_name: str,
        higher: bool,
    ) -> str:
        direction = "maximize" if higher else "minimize"
        prompt = f"""{program_md}

---
## Current best train.py
```python
{best_train_py}
```

---
## Experiment history
```
{progress_csv}
```

---
## Your task
Write the next version of train.py. Make ONE targeted change to {direction} `{metric_name}`.
- Study the history — do not repeat changes that already failed
- If the last 3+ rows all show improved=False, make a larger change (different model family, loss, or architecture)
- Keep `DATA_PATH = os.environ.get(...)` at the top
- Keep `BEST_VAL_{metric_name.upper()}: {{value:.6f}}` as the last printed line
- Return ONLY raw Python code — no markdown fences, no explanation."""

        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _strip_fences(resp.content[0].text.strip())

    # ------------------------------------------------------------------
    # PR creation via PyGithub
    # ------------------------------------------------------------------

    def _create_pr(
        self,
        progress_csv: Path,
        metric_name: str,
        best_value: float,
        total_experiments: int,
    ) -> str:
        try:
            from github import Auth, Github

            improved_count = sum(
                1 for row in csv.DictReader(open(progress_csv))
                if row.get("improved", "").lower() == "true"
            )

            auth = Auth.Token(self.github_token)
            gh   = Github(auth=auth)
            repo = gh.get_repo(self.github_repo)

            default_branch = repo.default_branch
            existing = list(repo.get_pulls(
                state="open",
                head=f"{self.github_repo.split('/')[0]}:{self.branch_name}",
            ))
            if existing:
                pr = existing[0]
                print(f"PR already exists: {pr.html_url}", flush=True)
                return pr.html_url

            pr = repo.create_pull(
                title=f"AutoTrain: best {metric_name}={best_value:.4f} "
                      f"({improved_count}/{total_experiments} improvements)",
                body=f"""## AutoTrain Results

| | |
|---|---|
| **Branch** | `{self.branch_name}` |
| **Metric** | `{metric_name}` |
| **Best value** | `{best_value:.6f}` |
| **Experiments** | {total_experiments} |
| **Improvements** | {improved_count} |

### What's in this branch
- `train.py` — best training script found
- `best_train.py` — same, explicit backup
- `progress.csv` — full experiment history
- `program.md` — instructions used by Claude

---
Generated by [AutoTrain](https://github.com/unionai/unionai-examples/tree/main/v2/tutorials/auto_train)
""",
                head=self.branch_name,
                base=default_branch,
            )
            print(f"PR created: {pr.html_url}", flush=True)
            return pr.html_url

        except Exception as e:
            print(f"PR creation failed (non-fatal): {e}", flush=True)
            return f"https://github.com/{self.github_repo}/tree/{self.branch_name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_from_program_md(program_text: str) -> None:
    """Parse ## Dependencies bullet list from program.md and pip install all packages."""
    m = re.search(r"## Dependencies\n(.*?)(?=\n##|\Z)", program_text, re.DOTALL)
    if not m:
        return
    packages = [
        line.lstrip("-").strip()
        for line in m.group(1).splitlines()
        if line.strip().lstrip("-").strip() and line.strip() != "none"
        and not line.strip().startswith("#")
        and not line.strip().startswith("These")
        and not line.strip().startswith("ResearchAgent")
        and not line.strip().startswith("If your")
    ]
    if packages:
        print(f"Installing dependencies from program.md: {packages}", flush=True)
        subprocess.run(["pip", "install", "--quiet"] + packages, check=False)


def _install_from_requires_comment(train_py: str) -> None:
    """Parse # REQUIRES: pkg1, pkg2 from the first 5 lines of train.py and install."""
    for line in train_py.splitlines()[:5]:
        m = re.match(r"#\s*REQUIRES:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            packages = [p.strip() for p in m.group(1).split(",") if p.strip()]
            if packages:
                print(f"Installing new packages from # REQUIRES: {packages}", flush=True)
                subprocess.run(["pip", "install", "--quiet"] + packages, check=False)
            break


def _parse_metric(stdout: str, metric_name: str) -> float | None:
    m = re.search(rf"BEST_VAL_{metric_name.upper()}:\s*([\d.]+)", stdout, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _is_better(new: float, best: float, higher: bool) -> bool:
    return (new > best + 1e-6) if higher else (new < best - 1e-6)


def _extract_model_name(train_py: str) -> str:
    for kw in ["XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor",
               "RandomForest", "MLP", "efficientnet", "resnet", "vit", "LSTM"]:
        if kw.lower() in train_py.lower():
            return kw
    return "unknown"


def _append_csv(
    path: Path, exp_id: int, train_py: str,
    metric_name: str, value: float, improved: bool, duration: float, notes: str,
) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            exp_id, _extract_model_name(train_py), metric_name,
            f"{value:.6f}", improved, duration, notes,
        ])


def _build_progress_plot(progress_csv: Path, metric_name: str, higher: bool) -> str:
    """Build a Plotly progress chart and return it as an HTML string."""
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
            xaxis_title="Experiment",
            yaxis_title=metric_name,
            height=400,
        )
        return fig.to_html(include_plotlyjs=True, full_html=False)

    except Exception as e:
        return f"<p>Plot unavailable: {e}</p>"


def _build_summary(
    progress_csv: Path, metric_name: str, best_value: float,
    total: int, branch_name: str, pr_url: str,
) -> str:
    improved_count = 0
    try:
        with open(progress_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("improved", "").lower() == "true":
                    improved_count += 1
    except Exception:
        pass
    return (
        f"AutoTrain complete\n"
        f"  Experiments : {total}\n"
        f"  Improvements: {improved_count}\n"
        f"  Best {metric_name}: {best_value:.6f}\n"
        f"  Branch      : {branch_name}\n"
        f"  PR          : {pr_url}"
    )


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[1:end])
    return text
