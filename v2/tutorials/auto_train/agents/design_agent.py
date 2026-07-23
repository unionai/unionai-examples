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
        starting_strategy = self._select_starting_strategy(profile)

        print(f"DesignAgent: metric={metric_name}  higher_is_better={higher_is_better}")
        print(f"  gpu={compute_config['gpu']}  memory={compute_config['memory']}  cpu={compute_config['cpu']}")
        print(f"  starting_strategy={starting_strategy!r}")

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
                starting_strategy=starting_strategy,
            )
        )
        (work_dir / "progress.csv").write_text(
            "experiment_id,description,model_name,metric_name,metric_value,improved,duration_s,commit,notes\n"
        )

        self._push_to_github(work_dir, folder_name, branch_name)
        return branch_name, folder_name, compute_config, packages

    # ------------------------------------------------------------------
    # Decide which tier of the modality ladder to start the baseline at
    # ------------------------------------------------------------------

    def _select_starting_strategy(self, profile: DataProfile) -> str:
        """Ask Claude to pick the right starting tier given the concrete data profile."""
        _BIO_KEYWORDS = ("dna", "rna", "protein", "genomic", "nucleotide",
                         "fasta", "amino", "splice", "bioinf", "genome", "transcript")
        is_biological = any(kw in profile.domain.lower() for kw in _BIO_KEYWORDS)

        ladders = {
            "tabular": (
                "Tier A: N<10k → GBM (LightGBM/XGBoost/CatBoost)\n"
                "Tier B: 10k≤N<100k → GBM first, then small MLP if it plateaus\n"
                "Tier C: N≥100k or heavy categoricals (>20 unique values, many cols) → GBM + deep tabular (TabNet, FT-Transformer)"
            ),
            "timeseries": (
                "Tier A: N<1k → classical ML on extracted features (rolling stats + FFT → XGBoost)\n"
                "Tier B: N≥1k AND L≤200 → 1D-CNN\n"
                "Tier C: N≥1k AND 200<L≤1000 AND N<50k → CNN-LSTM hybrid\n"
                "Tier D: N≥1k AND L>200 AND N≥50k → Transformer with patching (PatchTST-style)\n"
                "Tier E: N≥1k AND L>1000 AND N≥5k → Bidirectional LSTM/GRU"
            ),
            "sequence_biological": (
                "Tier A: short fixed-length (≤200 chars) AND N<5k → positional features + LightGBM or k-mer TF-IDF\n"
                "Tier B: any length AND N<5k → k-mer TF-IDF (ngram 3-6) + LightGBM\n"
                "Tier C: 1D-CNN on one-hot — only if k-mer clearly underperforms and N<5k\n"
                "Tier D: N≥5k → frozen DNA/protein language model + linear probe. "
                "Model choice: DNA sequences → zhihan1996/DNABERT-2-117M or InstaDeepAI/nucleotide-transformer-v2-100m; "
                "protein sequences → facebook/esm2_t6_8M_UR50D\n"
                "Tier E: N≥5k AND frozen probe plateau → two-phase fine-tuning (backbone.train())"
            ),
            "sequence_text": (
                "Tier A: N<2k → TF-IDF (word or char ngram) + LightGBM\n"
                "Tier B: 2k≤N<5k → TF-IDF + LightGBM, or small fine-tuned model (distilbert-base-uncased)\n"
                "Tier C: N≥5k → frozen pre-trained text transformer + linear probe. "
                "Model choice: general text → distilbert-base-uncased or roberta-base; "
                "domain-specific → try a domain-pretrained BERT if one exists (e.g. ProsusAI/finbert for finance)\n"
                "Tier D: N≥5k AND frozen probe plateau → full fine-tuning (all layers, LR=2e-5)"
            ),
            "image": (
                "Tier A: N<1k → frozen backbone, extract+cache embeddings once, sklearn classifier\n"
                "Tier B: 1k≤N<10k → lightweight ImageNet backbone (EfficientNet-B0, ResNet18), two-phase fine-tune\n"
                "Tier C: 10k≤N<50k → lightweight-to-medium backbone (EfficientNet-B2, ResNet34), full fine-tune\n"
                "Tier D: N≥50k → heavier backbone (EfficientNet-B4, ResNet50), full fine-tune + mixed precision"
            ),
        }

        if profile.modality == "sequence":
            ladder_key = "sequence_biological" if is_biological else "sequence_text"
        elif profile.modality in ladders:
            ladder_key = profile.modality
        else:
            ladder_key = "tabular"
        ladder = ladders[ladder_key]

        extra = ""
        if profile.modality == "sequence" and profile.avg_seq_length > 0:
            extra = f"\n- Avg sequence length: {profile.avg_seq_length} chars"
        if profile.modality in ("tabular", "timeseries"):
            extra = (
                f"\n- Features (F): {profile.num_features} "
                f"({profile.num_numeric_features} numeric, {profile.num_categorical_features} categorical)"
            )

        prompt = f"""You are an ML expert. Given this dataset profile, decide which starting tier the baseline experiment should use.

Dataset:
- Modality: {profile.modality}
- Sequence type: {"biological (DNA/RNA/protein)" if is_biological else "natural language text / other"}
- N (samples): {profile.num_samples:,}
- Domain: {profile.domain}
- Task: {profile.task_type}
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}{extra}

Available tiers:
{ladder}

Rules:
- Choose the tier most likely to produce a strong baseline directly.
- For biological sequences: k-mer and CNN are weak when sequences are long or the task requires context — go to Tier D (named DNA/protein model) when N≥5k.
- For NLP text: TF-IDF is only for N<5k; go to Tier C or D with a named text transformer for larger datasets.
- For image: tier is mainly N-driven; note domain-specific pretrained weights if relevant (medical, satellite, histology).
- For timeseries: use N and L; note multivariate if applicable.
- For tabular: use N and feature types; heavy categoricals push toward CatBoost or embeddings.
- The "approach" field MUST name a specific model (e.g. "zhihan1996/DNABERT-2-117M", "distilbert-base-uncased", "EfficientNet-B0") not just a method category.

Return ONLY a JSON object with exactly these fields:
{{
  "starting_tier": "<A|B|C|D|E>",
  "approach": "<specific model + method, e.g. frozen zhihan1996/DNABERT-2-117M linear probe, distilbert-base-uncased fine-tune, EfficientNet-B0 two-phase>",
  "reason": "<1-2 sentences referencing the actual numbers and why simpler tiers would waste experiments>"
}}"""

        resp = self.client.messages.create(
            model=self.MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _strip_fences(resp.content[0].text.strip())
        try:
            data = json.loads(raw)
            tier     = data.get("starting_tier", "A")
            approach = data.get("approach", "")
            reason   = data.get("reason", "")
            return (
                f"Tier {tier}: {approach}. "
                f"{reason} "
                f"Start here directly in experiment 0 — do not begin at a simpler tier."
            )
        except Exception:
            return ""

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

        # x-access-token works for both classic and fine-grained PATs; a bare
        # token-as-username fails for fine-grained (github_pat_*) tokens.
        repo_url = f"https://x-access-token:{self.github_token}@github.com/{self.github_repo}.git"
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(clone_dir)], check=True)
        subprocess.run(["git", "config", "user.email", GITHUB_EMAIL],    cwd=clone_dir, check=True)
        subprocess.run(["git", "config", "user.name",  GITHUB_USERNAME], cwd=clone_dir, check=True)

        # On an empty repo the first branch pushed becomes the repo's default,
        # which later breaks PR creation (base == head). Bootstrap a main
        # branch first so experiment branches always have a base to merge into.
        remote_heads = subprocess.run(
            ["git", "ls-remote", "--heads", "origin"],
            cwd=clone_dir, capture_output=True, text=True,
        )
        if not remote_heads.stdout.strip():
            subprocess.run(["git", "checkout", "-b", "main"], cwd=clone_dir, check=True)
            (clone_dir / "README.md").write_text(
                f"# AutoTrain experiments\n\nBranches in this repo are created by AutoTrain runs.\n"
            )
            subprocess.run(["git", "add", "README.md"], cwd=clone_dir, check=True)
            subprocess.run(["git", "commit", "-m", "AutoTrain: bootstrap main branch"], cwd=clone_dir, check=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=clone_dir, check=True)
            print("DesignAgent: bootstrapped main branch on empty repo", flush=True)

        subprocess.run(["git", "checkout", "-b", branch_name], cwd=clone_dir, check=True)

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
        starting_strategy: str = "",
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
            "starting_strategy": starting_strategy,
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
