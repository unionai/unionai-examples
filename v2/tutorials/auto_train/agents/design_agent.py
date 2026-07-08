"""
DesignAgent — runs once.

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
    if profile.modality in ("image", "sequence", "timeseries"):
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
# DesignAgent
# ---------------------------------------------------------------------------

class DesignAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str):
        self.github_repo  = github_repo
        self.github_token = os.environ.get("GITHUB_TOKEN", "")
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

        print(f"DesignAgent: metric={metric_name}  higher_is_better={higher_is_better}")
        print(f"  gpu={compute_config['gpu']}  memory={compute_config['memory']}  cpu={compute_config['cpu']}")

        train_py        = self._generate_train_py(profile, metric_name, time_budget_per_experiment_seconds)
        packages        = self._extract_packages(train_py)
        base_name       = self._generate_experiment_name(profile)
        folder_name     = f"{base_name}-{ts_label}"   # e.g. eurosat-efficientnet-multiclass-20260603-1430
        branch_name     = f"automl-{folder_name}"
        print(f"DesignAgent: experiment_folder={folder_name}  packages={packages}")

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

        (exp_dir / ".gitignore").write_text(
            "catboost_info/\n"
            "__pycache__/\n"
            "*.pyc\n"
            ".ipynb_checkpoints/\n"
        )

        subprocess.run(["git", "add", "."], cwd=clone_dir, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"AutoTrain: initial setup for {folder_name}"],
            cwd=clone_dir, check=True,
        )
        subprocess.run(["git", "push", "origin", branch_name], cwd=clone_dir, check=True)
        print(f"DesignAgent: pushed to {self.github_repo}/{branch_name}/{folder_name}", flush=True)
        shutil.rmtree(clone_dir)

    # ------------------------------------------------------------------
    # Generate initial train.py via Claude
    # ------------------------------------------------------------------

    def _generate_train_py(self, profile: DataProfile, metric_name: str, time_budget_per_experiment_seconds: float = 9000.0) -> str:
        """Generate a data-loading skeleton only — no model, no training loop.
        The research agent decides and implements the model in experiment 0."""
        seq_info = ""
        if profile.modality == "sequence" and profile.avg_seq_length > 0:
            seq_info = f" (avg sequence length: {profile.avg_seq_length:.0f} chars)"

        prompt = f"""Write only the data-loading section of a Python training script.
Do NOT implement any model, optimizer, loss function, or training loop.

## Dataset
- Modality: {profile.modality}
- Task type: {profile.task_type}
- Samples: {profile.num_samples:,}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Target column: {profile.target_column}
- Domain: {profile.domain}{seq_info}

## Rules
1. First two lines must be exactly:
   import os
   DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
2. Load and split data (80/20, stratified for classification):
   - Tabular  → pd.read_parquet(DATA_PATH); split into X_train/X_val/y_train/y_val
   - Sequence → pd.read_parquet(DATA_PATH); detect the sequence column (non-target string col);
                split into train_seqs/val_seqs/train_labels/val_labels as plain Python lists
   - Image    → detect native image size by sampling up to 10 images and taking the most common size:
                  from torchvision.datasets import ImageFolder
                  from PIL import Image
                  from collections import Counter
                  _tmp = ImageFolder(DATA_PATH)
                  _sizes = [Image.open(p).size for p, _ in _tmp.imgs[:10]]  # (width, height)
                  (native_w, native_h) = Counter(_sizes).most_common(1)[0][0]
                  n_channels = len(Image.open(_tmp.imgs[0][0]).getbands())
                  print(f"[DATA] Native image size: {{native_w}}x{{native_h}}, channels={{n_channels}}")
                Use native_h and native_w in ALL transforms — do NOT assume or hardcode any resolution.
                Handle grayscale: if n_channels == 1, add transforms.Grayscale(num_output_channels=1)
                and record img_channels=1 for the model to use later.
                For the skeleton, define train_transform and val_transform (val has no augmentation) but
                do NOT create DataLoaders yet — leave that for the model implementation step.
                Expose: dataset (full ImageFolder), train_idx, val_idx (index lists from stratified split),
                native_h, native_w, img_channels, num_classes.
3. For classification: encode labels to 0-based integers, print class mapping
4. Print a one-line dataset summary: sample counts, class distribution, and for images: native resolution + channels
5. num_workers=0 in any DataLoader
6. End with ONLY these three comment lines — no code after them:
   # TODO: implement model, training loop, and metric evaluation
   # Metric to optimize: {metric_name}
   # Final output line must be: print(f"BEST_VAL_{metric_name.upper()}: {{value:.6f}}")

Do NOT add any model imports, nn.Module, optimizer, loss, or training loop.
Return ONLY raw Python — no markdown fences, no explanation."""

        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return _strip_fences(resp.content[0].text.strip())

    # ------------------------------------------------------------------
    # Detect required packages from the generated train.py
    # ------------------------------------------------------------------

    def _extract_packages(self, train_py: str) -> list[str]:
        """train.py is now a skeleton with no model — no extra packages to detect.
        The research agent installs any needed packages via shell during experiment 0."""
        return []

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
        prompts_dir = Path(__file__).parent / "prompts"

        modality_file = {
            "tabular":    "strategy_tabular.md",
            "image":      "strategy_image.md",
            "timeseries": "strategy_timeseries.md",
            "sequence":   "strategy_sequence.md",
        }.get(profile.modality, "strategy_tabular.md")

        modality_ctx: dict = {
            "tab_n":   profile.num_samples,
            "tab_f":   profile.num_features,
            "tab_num": profile.num_numeric_features,
            "tab_cat": profile.num_categorical_features,
            "img_n":   profile.num_samples,
            "ts_n":    profile.num_samples,
            "ts_l":    profile.num_features,
        }
        modality_section = (prompts_dir / modality_file).read_text().format(**modality_ctx)

        base_ctx = {
            "direction":         "Maximize" if higher_is_better else "Minimize",
            "metric_name":       metric_name,
            "metric_upper":      metric_name.upper(),
            "task_type":         profile.task_type,
            "higher_is_better":  str(higher_is_better).lower(),
            "modality":          profile.modality,
            "num_samples":       profile.num_samples,
            "num_features":      profile.num_features,
            "num_numeric":       profile.num_numeric_features,
            "num_categorical":   profile.num_categorical_features,
            "num_classes":       profile.num_classes,
            "class_distribution": json.dumps(profile.class_distribution),
            "missing_rate":      profile.missing_rate,
            "target_column":     profile.target_column,
            "domain":            profile.domain,
            "gpu":               compute_config["gpu"],
            "memory":            compute_config["memory"],
            "cpu":               compute_config["cpu"],
            "disk":              compute_config["disk"],
            "time_budget":       time_budget,
            "max_experiments":   max_experiments,
            "modality_section":  modality_section,
        }
        return (prompts_dir / "base.md").read_text().format(**base_ctx)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[1:end])
    return text
