"""
ArchitectureAgent — runs once.

1. Reads DataProfile
2. Selects compute tier (GPU) based on data size and modality
3. Uses Claude to generate the initial train.py
4. Uses Claude to generate a descriptive experiment folder name
5. Detects required pip packages from the generated train.py
6. Writes program.md with instructions for the research loop
7. Clones the GitHub repo, creates a branch, pushes train.py + program.md
8. Returns (branch_name, experiment_folder, compute_config, packages)
   compute_config and packages are passed back as task outputs so run.py
   can build the correct GPU image and resource spec for ResearchAgent.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import anthropic

from agents.data_agent import DataProfile

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL    = "parnianzargham@gmail.com"


# ---------------------------------------------------------------------------
# Compute tier selection
# ---------------------------------------------------------------------------

def _select_compute_tier(profile: DataProfile) -> dict:
    """
    Decide GPU / memory requirements based on data size and modality.
    Always GPU — just varies the memory and core count.
    """
    if profile.modality in ("image", "sequence"):
        return {
            "gpu": "T4:1",
            "memory": "32Gi",
            "cpu": 8,
            "disk": "100Gi",
        }
    if profile.num_samples > 50_000:
        return {
            "gpu": "T4:1",
            "memory": "16Gi",
            "cpu": 8,
            "disk": "50Gi",
        }
    # Small tabular — lighter GPU allocation
    return {
        "gpu": "T4:1",
        "memory": "8Gi",
        "cpu": 4,
        "disk": "20Gi",
    }




# ---------------------------------------------------------------------------
# ArchitectureAgent
# ---------------------------------------------------------------------------

class ArchitectureAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str, github_token: str):
        self.github_repo  = github_repo
        self.github_token = github_token
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Ask Claude which metric to optimise
    # ------------------------------------------------------------------

    def _select_metric(self, profile: DataProfile) -> tuple[str, bool]:
        """Ask Claude to choose the best evaluation metric for this task."""
        prompt = f"""You are an ML expert. Choose the single best evaluation metric for this task.

Dataset:
- Modality: {profile.modality}
- Task type: {profile.task_type}
- Samples: {profile.num_samples:,}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Domain: {profile.domain}

Return a JSON object with exactly these two fields:
{{
  "metric_name": "<snake_case name, e.g. roc_auc, macro_f1, rmse, accuracy, mse, r2, mcc>",
  "higher_is_better": true or false
}}

Return ONLY the JSON, no explanation."""

        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _strip_fences(resp.content[0].text.strip())
        data = json.loads(raw)
        metric_name      = data["metric_name"].lower().replace(" ", "_")
        higher_is_better = bool(data["higher_is_better"])
        return metric_name, higher_is_better

    def run(
        self,
        profile: DataProfile,
        time_budget_per_experiment_seconds: float = 300.0,
        max_experiments: int = 20,
    ) -> tuple[str, str, dict, list[str]]:
        """
        Generate train.py + program.md, push to a new GitHub branch.
        Returns (branch_name, experiment_folder, compute_config, packages).
        packages is the pip list for the research GPU image.
        compute_config has gpu/memory/cpu/disk specs for the research env.
        """
        import time
        ts       = int(time.time())
        ts_label = time.strftime("%Y%m%d-%H%M", time.gmtime(ts))
        work_dir = Path(f"/tmp/automl_arch_{ts}")
        work_dir.mkdir(parents=True, exist_ok=True)

        metric_name, higher_is_better = self._select_metric(profile)
        compute_config = _select_compute_tier(profile)

        print(f"ArchitectureAgent: metric={metric_name}  higher_is_better={higher_is_better}")
        print(f"  gpu={compute_config['gpu']}  memory={compute_config['memory']}  cpu={compute_config['cpu']}")

        train_py        = self._generate_train_py(profile, metric_name)
        packages        = self._extract_packages(train_py)
        base_name       = self._generate_experiment_name(profile)
        folder_name     = f"{base_name}-{ts_label}"   # e.g. eurosat-efficientnet-multiclass-20260603-1430
        branch_name     = f"automl-{folder_name}"
        print(f"ArchitectureAgent: experiment_folder={folder_name}  packages={packages}")

        (work_dir / "train.py").write_text(train_py)
        (work_dir / "program.md").write_text(
            self._write_program_md(
                profile, metric_name, higher_is_better,
                time_budget_per_experiment_seconds, max_experiments,
                packages, compute_config, branch_name,
            )
        )
        (work_dir / "progress.csv").write_text(
            "experiment_id,description,model_name,metric_name,metric_value,improved,duration_s,commit,notes\n"
        )

        self._push_to_github(work_dir, folder_name, branch_name)
        return branch_name, folder_name, compute_config, packages

    # ------------------------------------------------------------------
    # Generate a descriptive experiment folder name via Claude
    # ------------------------------------------------------------------

    def _generate_experiment_name(self, profile: DataProfile) -> str:
        prompt = f"""Generate a short folder name for this ML experiment.
The name should clearly reflect the dataset/domain so someone browsing the repo
immediately knows what data it runs on.

Dataset info:
- Domain / source: {profile.domain}
- Modality: {profile.modality}
- Task: {profile.task_type}
- Samples: {profile.num_samples}
- Target column: {profile.target_column}

Rules:
- Start with the dataset or domain name (e.g. eurosat, titanic, brain-tumor, iris)
- Append the model family and task type
- lowercase, hyphens only, 3–5 words total
- Good examples: eurosat-efficientnet-multiclass, titanic-lgbm-binary, brain-tumor-effnet-4class

Return ONLY the folder name, nothing else."""
        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip().lower()
        name = re.sub(r"[^a-z0-9-]", "-", raw).strip("-")
        return name or "automl-experiment"

    # ------------------------------------------------------------------
    # Push to GitHub
    # ------------------------------------------------------------------

    def _push_to_github(self, work_dir: Path, folder_name: str, branch_name: str) -> None:
        """Clone repo, create branch_name, copy arch files into folder_name/, commit and push."""
        import time as _time
        clone_dir = Path(f"/tmp/automl_arch_git_{int(_time.time())}")
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        repo_url = f"https://{self.github_token}@github.com/{self.github_repo}.git"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.email", GITHUB_EMAIL],    cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.name",  GITHUB_USERNAME], cwd=clone_dir, check=True)

        exp_dir = clone_dir / folder_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["train.py", "program.md", "progress.csv"]:
            src = work_dir / fname
            if src.exists():
                shutil.copy(src, exp_dir / fname)

        subprocess.run(["git", "add", "."], cwd=clone_dir, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"AutoTrain: initial setup for {folder_name}"],
            cwd=clone_dir, check=True,
        )
        subprocess.run(["git", "push", "origin", branch_name], cwd=clone_dir, check=True)
        print(f"ArchitectureAgent: pushed to {self.github_repo}/{branch_name}/{folder_name}", flush=True)
        shutil.rmtree(clone_dir)

    # ------------------------------------------------------------------
    # Generate initial train.py via Claude
    # ------------------------------------------------------------------

    def _generate_train_py(self, profile: DataProfile, metric_name: str) -> str:
        prompt = f"""Write a complete, self-contained Python training script.

## Dataset
- Modality: {profile.modality}
- Task type: {profile.task_type}
- Samples: {profile.num_samples}
- Features: {profile.num_features}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Target column: {profile.target_column}
- Domain: {profile.domain}

## Requirements
1. First two lines must be exactly:
   import os
   DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
2. Tabular data: load DATA_PATH as a parquet file. Image data: load as ImageFolder directory.
3. 80/20 train/val split (stratified for classification)
4. Best starting model for the data size and task — follow these rules strictly:
   - Modality is "image" → ALWAYS use a pretrained CNN backbone from timm, NEVER tree-based models
     (XGBoost / LightGBM are for tabular data only)
     Choose the backbone based on dataset size and a single T4 GPU (16Gi VRAM):
     - Pick a lightweight backbone for <50k images, heavier one for larger datasets
     - Justify your choice in a comment at the top of the script
   - Tabular <10k samples → XGBoost or LightGBM
   - Tabular >10k samples → LightGBM
5. Compute `{metric_name}` on the validation set and print it as the very last line:
   BEST_VAL_{metric_name.upper()}: {{value:.6f}}
   Implement the metric yourself using sklearn or the relevant library — do not skip this.
6. You may use any pip-installable package appropriate for the task
7. No hardcoded paths. Under 150 lines.

Return ONLY raw Python — no markdown fences, no explanation."""

        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        return _strip_fences(resp.content[0].text.strip())

    # ------------------------------------------------------------------
    # Detect required packages from the generated train.py
    # ------------------------------------------------------------------

    def _extract_packages(self, train_py: str) -> list[str]:
        """Ask Claude which pip packages the script needs (excluding stdlib)."""
        prompt = f"""List the pip package names needed to run this Python script.
Only include packages that must be pip installed — exclude Python stdlib modules.
Return one package name per line, no versions, no explanation, no bullet points.

```python
{train_py}
```"""
        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        return [
            p.strip().lstrip("-").strip()
            for p in raw.splitlines()
            if p.strip() and not p.startswith("#")
        ]

    # ------------------------------------------------------------------
    # Write program.md
    # ------------------------------------------------------------------

    def _write_program_md(
        self,
        profile: DataProfile,
        metric_name: str,
        higher_is_better: bool,
        time_budget: float,
        max_experiments: int,
        packages: list[str],
        compute_config: dict,
        branch_name: str,
    ) -> str:
        direction = "maximize" if higher_is_better else "minimize"
        recs = "\n".join(f"- {r}" for r in profile.recommendations) if profile.recommendations else "- none"
        deps = "\n".join(f"- {p}" for p in packages) if packages else "- none"

        return f"""# AutoTrain Research Program

## Role
You are an AI researcher. Your job is to come up with the best possible training script for this task. Each iteration, study the experiment history and propose a meaningful improvement — do not make random changes, think carefully about what is likely to help.

## Goal
{direction.capitalize()} `{metric_name}` on a `{profile.task_type}` task.
higher_is_better: {str(higher_is_better).lower()}

Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, loss function, data augmentation, regularization, model size, etc.

The metric must be printed as the final line of every run:
```
BEST_VAL_{metric_name.upper()}: {{value:.6f}}
```

## Dataset
- Modality: {profile.modality}
- Task: {profile.task_type}
- Samples: {profile.num_samples:,}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Target: `{profile.target_column}`
- Domain: {profile.domain}

## Compute
- GPU: {compute_config['gpu']}
- Memory: {compute_config['memory']}
- CPU: {compute_config['cpu']} cores
- Time budget per experiment: {time_budget:.0f}s
- Max experiments: {max_experiments}

## Experiment loop

Run up to {max_experiments} experiments (0-indexed). You decide when to stop — stop only when you are confident further improvement is unlikely.

Steps A–E below are **mandatory for every experiment, in order**.

---

### A. Prepare train.py

**Experiment 0 (baseline):**
Run `train.py` exactly as provided — do NOT change model architecture, hyperparameters, or training logic.
If it crashes with an import error or syntax error, fix only that error and re-run. Keep fixing crash-only errors until the script produces a metric output. The description for experiment 0 is always `"baseline"`.

**Experiments 1 and later:**
Make exactly one change to improve `{metric_name}`: try a different model architecture, optimizer, learning rate schedule, regularization, data augmentation, batch size, etc.
Study `progress.csv` first — do not repeat a change that already failed. If the last 3 experiments all failed to improve, make a bolder change.

---

### B. Run

```bash
python train.py 2>&1 | tee /tmp/train_out.txt
```

If `python` is not found, use `python3`. `DATA_PATH` is already set in the environment.

---

### C. Write to progress.csv immediately after the run (MANDATORY — do not skip)

Parse the metric from the output. Look for this exact line anywhere in the output:
```
BEST_VAL_{metric_name.upper()}: <value>
```

Open `progress.csv` in **append** mode and write one row:
```
<exp_id>,<description>,<model_name>,{metric_name},<value>,<improved>,<duration_s>,,<notes>
```

- `exp_id`: integer, 0-indexed
- `description`: `"baseline"` for exp 0; one-line description of the single change for exp 1+
- `model_name`: main model class (e.g. `LGBMClassifier`, `EfficientNet`)
- `metric_value`: the float value, or leave empty if the script crashed before printing it
- `improved`: `True` if this run's value is {"strictly greater than" if higher_is_better else "strictly less than"} the previous best, else `False` (always `True` for exp 0)
- `duration_s`: wall-clock seconds
- `commit`: leave empty for now — fill it in after step D
- `notes`: empty on success; one-line error summary if the script crashed

**Even if the run crashed, write the row.** Do not skip this step.

---

### D. Keep or discard

- If `improved=True`: `cp train.py best_train.py`
- If `improved=False` and exp > 0: `cp best_train.py train.py`

---

### E. Commit and fill in the commit hash

```bash
git add train.py best_train.py progress.csv
git commit -m "exp <exp_id>: {metric_name}=<value> [KEEP/DISCARD]"
git push origin {branch_name}
```

After committing, get the short hash:
```bash
git rev-parse --short HEAD
```

Go back to `progress.csv` and fill in the `commit` column for the row you just wrote.

---

Repeat from A for the next experiment.

## Constraints
- First two lines of `train.py` must always be:
  ```python
  import os
  DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
  ```
- Final printed line must be exactly: `BEST_VAL_{metric_name.upper()}: {{value:.6f}}`
- Install new pip packages with `pip install <pkg>` in the shell — do not put pip calls inside `train.py`
- Only edit `train.py`, `best_train.py`, and `progress.csv`

## Pre-installed packages
{deps}

## progress.csv
The file already exists with the header. Append one row per experiment — never rewrite the whole file.
"""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[1:end])
    return text
