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
# Metric selection
# ---------------------------------------------------------------------------

def _select_metric(profile: DataProfile) -> tuple[str, bool]:
    """Returns (metric_name, higher_is_better)."""
    if profile.task_type == "regression":
        return "rmse", False
    if profile.class_distribution:
        counts = list(profile.class_distribution.values())
        if counts and max(counts) / max(min(counts), 1) > 3:
            return "macro_f1", True
    if profile.num_classes == 2:
        return "roc_auc", True
    return "macro_f1", True


# ---------------------------------------------------------------------------
# ArchitectureAgent
# ---------------------------------------------------------------------------

class ArchitectureAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str, github_token: str):
        self.github_repo  = github_repo
        self.github_token = github_token
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
        work_dir = Path(f"/tmp/automl_arch_{int(time.time())}")
        work_dir.mkdir(parents=True, exist_ok=True)

        metric_name, higher_is_better = _select_metric(profile)
        compute_config = _select_compute_tier(profile)

        print(f"ArchitectureAgent: metric={metric_name}")
        print(f"  gpu={compute_config['gpu']}  memory={compute_config['memory']}  cpu={compute_config['cpu']}")

        train_py        = self._generate_train_py(profile, metric_name)
        packages        = self._extract_packages(train_py)
        experiment_name = self._generate_experiment_name(profile)
        print(f"ArchitectureAgent: experiment_folder={experiment_name}  packages={packages}")

        (work_dir / "train.py").write_text(train_py)
        (work_dir / "program.md").write_text(
            self._write_program_md(
                profile, metric_name, higher_is_better,
                time_budget_per_experiment_seconds, max_experiments,
                packages, compute_config,
            )
        )
        (work_dir / "progress.csv").write_text(
            "experiment_id,model_name,metric_name,metric_value,improved,duration_s,notes\n"
        )

        branch_name = self._push_to_github(work_dir, experiment_name)
        return branch_name, experiment_name, compute_config, packages

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

    def _push_to_github(self, work_dir: Path, folder_name: str) -> str:
        """
        Clone repo, create branch automl-{folder_name}-{ts},
        copy arch files into {folder_name}/, commit and push.
        Returns branch_name.
        """
        import time as _time
        ts          = int(_time.time())
        branch_name = f"automl-{folder_name}-{ts}"
        clone_dir   = Path(f"/tmp/automl_arch_git_{ts}")
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
        return branch_name

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
1. First line: DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
2. Tabular data: load DATA_PATH as a parquet file. Image data: load as ImageFolder directory.
3. 80/20 train/val split (stratified for classification)
4. Best starting model for the data size and task:
   - Tabular <10k samples → XGBoost or LightGBM
   - Tabular >10k samples → LightGBM
   - Images → EfficientNet-B0 from timm (pretrained=True)
5. Last printed line must be exactly: BEST_VAL_{metric_name.upper()}: {{value:.6f}}
6. You may use any pip-installable package appropriate for the task
7. No hardcoded paths. Under 120 lines.

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
    ) -> str:
        direction = "maximize" if higher_is_better else "minimize"
        recs = "\n".join(f"- {r}" for r in profile.recommendations) if profile.recommendations else "- none"
        deps = "\n".join(f"- {p}" for p in packages) if packages else "- none"

        return f"""# AutoTrain Research Program

## Objective
{direction.capitalize()} `{metric_name}` on a `{profile.task_type}` task.

Required output format (never change this line):
```
BEST_VAL_{metric_name.upper()}: {{value:.6f}}
```

## Data
- Modality: {profile.modality}
- Samples: {profile.num_samples:,}
- Features: {profile.num_features}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Target column: `{profile.target_column}`
- Quality score: {profile.quality_score:.2f}

### Data recommendations
{recs}

## Compute
- GPU: {compute_config['gpu']}
- Memory: {compute_config['memory']}
- CPU: {compute_config['cpu']} cores
- Disk: {compute_config['disk']}
- Time budget per experiment: {time_budget:.0f}s
- Max experiments: {max_experiments}

## Dependencies
ResearchAgent installs these before each experiment via `pip install`.
{deps}

If your next train.py change requires a new package not listed above,
annotate it on the first line: `# REQUIRES: pkg1, pkg2`
ResearchAgent will install it before running.

## Rules
1. Modify ONLY `train.py` — one targeted change per iteration
2. `DATA_PATH = os.environ.get("DATA_PATH", ...)` must stay at the top
3. Always end with exactly: `BEST_VAL_{metric_name.upper()}: {{value:.6f}}`
4. Annotate new packages as `# REQUIRES: pkg` — do NOT call pip install inside train.py
5. Script must be self-contained and runnable

## What to try (in priority order)
1. Learning rate, LR schedule (cosine, step, warmup)
2. n_estimators, max_depth, min_child_weight, subsample, colsample_bytree
3. Regularization — alpha, lambda, dropout, weight_decay
4. Class weights or focal loss for imbalanced classes
5. Feature engineering or selection (tabular only)
6. Augmentation intensity (images only)
7. Model family upgrade if consistently stuck:
   XGBoost → LightGBM → MLP (tabular)
   EfficientNet-B0 → B2 → B4 (images)

## Progress
See `progress.csv` for the full experiment history. Check it before proposing a change.
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
