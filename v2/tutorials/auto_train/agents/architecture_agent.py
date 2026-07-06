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

        train_py        = self._generate_train_py(profile, metric_name, time_budget_per_experiment_seconds)
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
        print(f"ArchitectureAgent: pushed to {self.github_repo}/{branch_name}/{folder_name}", flush=True)
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
        direction = "maximize" if higher_is_better else "minimize"

        is_timeseries = profile.modality == "timeseries"
        is_tabular    = profile.modality == "tabular"
        is_image      = profile.modality == "image"

        tab_n    = profile.num_samples
        tab_f    = profile.num_features
        tab_num  = profile.num_numeric_features
        tab_cat  = profile.num_categorical_features
        # High cardinality categorical: any cat feature column with potentially >20 unique values
        # (exact cardinality not tracked here; use tab_cat > 0 as signal and note in section)
        heavy_cat = tab_cat > 0

        tabular_section = f"""
**Tabular strategy** (N={tab_n:,} samples, F={tab_f} features — {tab_num} numeric, {tab_cat} categorical):

Decision ladder:
1. **N < 10,000 → Gradient boosting** (LightGBM / XGBoost / CatBoost). Deep learning rarely beats GBMs at this scale. *This dataset fits here if N={tab_n:,} < 10,000.*
2. **10,000 ≤ N < 100,000 → GBM first**, then try a small MLP (2–3 hidden layers) only if GBM has clearly plateaued and you need to beat it.
3. **N ≥ 100,000 → GBM still competitive**; deep tabular models (TabNet, FT-Transformer, MLP with embeddings) become viable. Try both — GBM often still wins or ties.
4. **Heavy categorical features** (many columns with high cardinality, >20 unique values):
   → CatBoost (handles natively without encoding) or MLP with learned entity embeddings per category.
5. **Mostly numeric + low-cardinality categoricals** (<20 unique values per column):
   → LightGBM / XGBoost with one-hot or ordinal encoding. Simpler and faster.

Feature engineering to consider before switching models:
- **Numeric**: log/sqrt transform for right-skewed features; polynomial interactions (degree 2) for small F; binning high-range features.
- **Categorical**: target encoding for high-cardinality (>20 unique); one-hot for low-cardinality.
- **Missing values**: median impute for numeric, mode/constant for categorical; add a binary missingness-indicator flag for columns with >5% missing.
- **Feature selection**: after the first GBM fit, drop features with zero importance. Re-check with permutation importance if tree importance is misleading.
- **Cross-validation**: for N < 10,000 prefer 5-fold stratified CV (more reliable than a single 80/20 split). Report mean ± std across folds.
""" if is_tabular else ""

        img_n = profile.num_samples

        image_section = f"""
**Image strategy** (N={img_n:,} samples, native resolution and channels detected at runtime by the data skeleton):

Variables already in scope from the skeleton: `dataset` (ImageFolder), `train_idx`, `val_idx`, `native_h`, `native_w`, `img_channels`, `num_classes`, `train_transform`, `val_transform`. Build DataLoaders with `SubsetRandomSampler(train_idx)` / `SubsetRandomSampler(val_idx)` — do NOT re-split.

Scale-based strategy:
- **N < 1,000** — freeze backbone completely. Extract embeddings once (FP16, no_grad, batch=256), cache to disk. Train a sklearn LogisticRegression or SVC on the cached embeddings. No backward pass through the backbone.
- **1,000 ≤ N < 10,000** — lightweight ImageNet-pretrained backbone (EfficientNet-B0, MobileNetV3-Small, ResNet18). Two-phase training: Phase 1 — freeze backbone, train linear head (10–15 epochs, LR=1e-3). Phase 2 — unfreeze all, cosine LR schedule (backbone LR=1e-4, head LR=1e-3, 20–30 epochs).
- **10,000 ≤ N < 50,000** — lightweight to medium backbone (EfficientNet-B0/B2, ResNet34). Full fine-tuning from the start or after a short 5-epoch frozen warm-up. Aggressive augmentation (random crop, flip, color jitter, random erasing).
- **N ≥ 50,000** — heavier backbone (EfficientNet-B4, ResNet50). Full fine-tuning with mixed precision (`torch.cuda.amp.autocast`) to fit larger batches in VRAM.

Backbone selection principles:
- Load from `timm` (`timm.create_model(name, pretrained=True, num_classes=num_classes)`). Prefer small backbones for small N — EfficientNet-B0 (~5M params) generalizes better than B4 when data is limited.
- For domain-specific images (medical, satellite, histology): check if a domain-pretrained model exists (e.g. BiT, RadImageNet for medical) — it will transfer better than standard ImageNet weights.
- Do NOT use backbones >50M parameters with N<10k — the head overfits to noise even with a frozen backbone.
- Always use `native_h` and `native_w` from the skeleton in ALL transforms — never hardcode a resolution.

Resolution and channels:
- Use `transforms.Resize((native_h, native_w))` or `transforms.Resize(min(native_h, native_w))` + `CenterCrop`.
- If `img_channels == 1` (grayscale): add `transforms.Grayscale(num_output_channels=1)` to both transforms and set the backbone's first conv `in_channels=1` (or duplicate the channel: `transforms.Grayscale(num_output_channels=3)` and use standard weights).
- Normalize with ImageNet mean/std when using ImageNet-pretrained backbones: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

Augmentation (training only — val_transform must be resize + normalize only):
- Standard: `RandomHorizontalFlip`, `RandomCrop` with padding, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.
- For small N: also add `RandomRotation(15)` and `RandomErasing(p=0.25)`.
- Do NOT apply random augmentation to the validation set.

Class imbalance: use `WeightedRandomSampler` in the training DataLoader (sample weight = inverse class frequency) rather than oversampling the dataset. This avoids val leakage and is simpler than SMOTE for images.
""" if is_image else ""

        # For time series: L = num_features (timesteps per sample), C typically = 1 for univariate
        ts_n, ts_l = profile.num_samples, profile.num_features

        ts_section = f"""
**Time series strategy** (N={ts_n:,} samples, L={ts_l} timesteps/features per sample, C=1 for univariate — increase C if channels are stored as separate column groups):

Work through this ladder in order — do NOT jump straight to a transformer:

1. **N < 1,000 → Classical ML on extracted features**: rolling mean/std/min/max per window + FFT coefficients → XGBoost or Random Forest. Deep nets overfit at this scale regardless of architecture.
2. **N ≥ 1,000 AND L ≤ 200 → 1D-CNN** (3–4 conv blocks + global average pool + FC head). Cheap, fast, sufficient receptive field for short windows. *This dataset falls here if L={ts_l} ≤ 200.*
3. **N ≥ 1,000 AND 200 < L ≤ 1,000**:
   - N ≥ 50k → Transformer with patching (PatchTST-style: divide series into non-overlapping patches, add positional encoding, apply attention)
   - N < 50k → 1D-CNN optionally with an LSTM head (CNN-LSTM hybrid)
4. **N ≥ 1,000 AND L > 1,000**:
   - N ≥ 50k → Transformer with patching
   - N ≥ 5k → Bidirectional LSTM/GRU
   - N < 5k → Chunk series into patches first, then CNN (not enough data for deep LSTM or Transformer)
5. **Multivariate (C > 1)**: prefer CNN or Transformer over plain LSTM — channel mixing is easier with conv/attention than with flat-concatenated LSTM input.
6. **Streaming/online inference**: override all of the above → LSTM/GRU (CNN and Transformer require the full window upfront).

**Reshape note** — X_train from the skeleton is shape (N, L) flat. Before passing to a model:
- 1D-CNN: `X.reshape(N, C, L)` → `nn.Conv1d(in_channels=C, ...)`
- LSTM/GRU: `X.reshape(N, L, C)` → `nn.LSTM(input_size=C, ...)`
- Classical ML: use flat (N, L) directly, or compute features first.
""" if is_timeseries else ""

        return f"""# AutoTrain Research Program

## Role
You are an AI researcher. Your job is to find the best possible training script for this task. You choose the model, architecture, and training strategy. Each iteration, study the experiment history and make a meaningful, reasoned improvement.

## Goal
{direction.capitalize()} `{metric_name}` on a `{profile.task_type}` task.
higher_is_better: {str(higher_is_better).lower()}

Modify `train.py` — this is the only file you edit. Everything is fair game: model choice, architecture, optimizer, hyperparameters, training loop, loss function, data augmentation, regularization. The metric must be the final printed line of every run:
```
BEST_VAL_{metric_name.upper()}: {{value:.6f}}
```

## Dataset
- Modality: {profile.modality}
- Task: {profile.task_type}
- Samples (N): {profile.num_samples:,}
- Features (F): {profile.num_features} ({profile.num_numeric_features} numeric, {profile.num_categorical_features} categorical)
- Classes: {profile.num_classes}
- Class distribution: {json.dumps(profile.class_distribution)}
- Missing rate: {profile.missing_rate:.1%}
- Target: `{profile.target_column}`
- Domain: {profile.domain}

## Compute
- GPU: {compute_config['gpu']} (16 GB VRAM)
- RAM: {compute_config['memory']}
- CPU: {compute_config['cpu']} cores
- Disk: {compute_config['disk']}
- Time budget per experiment: {time_budget:.0f}s
- Max experiments: {max_experiments}

## Hard constraints (never override — apply to every experiment)
- `num_workers=0` in ALL DataLoaders — worker processes hang in containers
- No `pip install` calls inside `train.py` — install in the shell before running
- Only edit `train.py`, `best_train.py`, and `progress.csv`
- First two lines of `train.py` must always be:
  ```python
  import os
  DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")
  ```
- **Never hardcode any data path** — not even as a fallback. Do NOT write `try: pd.read_csv(DATA_PATH) except: pd.read_csv('/some/hardcoded/path')`. The `DATA_PATH` env var is always set by the runner; if reading fails, raise the error so it is visible.
- Final printed line must be exactly: `BEST_VAL_{metric_name.upper()}: {{value:.6f}}`
- **No threshold calibration on the validation set** — do NOT use `scipy.optimize`, `differential_evolution`, or any search method to find per-class decision thresholds by fitting to val labels. This leaks val labels into the decision boundary and inflates the reported metric. Use argmax over softmax/sigmoid outputs as-is, or apply a single global threshold chosen on held-out calibration data that is not part of the reported val split.
- **No double normalization** — apply exactly one normalization pass. Do NOT z-score per-sample and then z-score globally (or vice versa). Pick one: per-sample standardization OR dataset-level standardization, and apply it once.
- **CatBoost `train_dir`** — always pass `train_dir='/tmp/catboost_info'` when constructing any CatBoost model. Without this, CatBoost writes a `catboost_info/` directory into the experiment folder which gets committed to git.

## Model selection principles
Reason from these constraints — do not rely on a fixed list:

**Size**: choose models that fit within the compute budget. Estimate before committing:
- GPU VRAM for the model + activations + optimizer states must stay under 14 GB
- For pretrained transformers: parameter count × 4 bytes (FP32) or × 2 bytes (FP16) for inference
- Embedding extraction at batch_size=256, FP16: 50M-param model ≈ 2–4 min for 50k sequences on T4
- Full fine-tuning (Phase 2): batch_size=16–32, forward+backward ≈ 5–10s/batch for 50M-param model

**Data size vs model capacity**:
- Tabular (modality=tabular): GBM first, deep tabular only when N is large and GBM has plateaued — see Tabular strategy section below
- Time series (modality=timeseries): do NOT use LightGBM as-is — see Time series strategy section below
- Image (modality=image): transfer learning from pretrained backbone, scale of fine-tuning depends on N — see Image strategy section below
{tabular_section}{image_section}

**Sequence strategy — work through this decision ladder in order, do NOT jump straight to a transformer:**

1. **Positional tabular features** (if sequence is fixed-length and short, e.g. ≤200 chars):
   Split each character position into a separate categorical feature column → LightGBM/XGBoost.
   e.g. for a 60-char DNA sequence: `df[[f'pos_{{i}}' for i in range(60)]] = df['seq'].apply(list, result_type='expand')`
   This is often the strongest approach for short fixed-length sequences (DNA motifs, splice sites, short peptides).
   Try this FIRST — a 0.90+ result here means you do not need a transformer.

2. **k-mer frequency features** (works for any sequence length):
   `CountVectorizer(analyzer='char', ngram_range=(3, 6))` → LightGBM or logistic regression.
   Fast, no GPU needed. Add this as experiment 1 if positional features don't reach a satisfying ceiling.

3. **CNN on one-hot encoded sequences** (if the above plateau):
   One-hot encode each position → 1D CNN. Captures local motifs better than k-mers without a pretrained model.
   Good when sequences are variable-length or when k-mer features miss positional context.

4. **Frozen domain-specific transformer + linear probe** (only when N≥5k and CNN also plateaus):
   Extract CLS/mean-pool embeddings once at FP16 with `torch.no_grad()`, cache to disk, train a linear head.
   Choose a model pretrained on the same domain (DNA, protein, text, etc.).

5. **Two-phase fine-tuning** (only when frozen probe plateaus AND N≥5k):
   Phase 1: frozen backbone, cache embeddings, train head (3–5 epochs).
   Phase 2: `backbone.train().float()`, new DataLoader of raw sequences, backbone LR=1e-5 / head LR=1e-4, batch=16–32.

Start at step 1. Skip to a later step only if the current approach has clearly plateaued. Jumping to step 4/5 with small data typically underperforms step 1 or 2.
{ts_section}
**Domain fit**: prefer models pretrained on data similar to the task domain. A model pretrained on a related domain will transfer better even if it has fewer parameters than a large general model.

**Compatibility constraints** (environment limitations — hard failures if violated):
- No models that require flash-attention (causes ImportError at load time)
- No models over ~200M parameters — slower than the time budget allows for embedding extraction
- Never use `ignore_mismatched_sizes=True` — it silently randomly-initializes mismatched layers and destroys pretrained weights
- Always pass `config=config` to `AutoModel.from_pretrained` — without it, any config patches via `object.__setattr__` are silently ignored
- If `AutoModel` raises a size-mismatch error (MLM/causal checkpoint with extra head weights): use the task-specific class (e.g. `AutoModelForMaskedLM`) and extract the encoder sub-module (`.bert`, `.roberta`, `.esm`, `.distilbert`, etc. — inspect `model.__class__.__name__` if unsure)

**Class imbalance**: when the largest class is >2× the smallest, use focal loss with alpha = inverse class frequency (1/count, normalized to sum to 1). Do NOT use raw class frequencies as alpha — that down-weights minority classes.

## Experiment loop

The Python runner manages the loop: it calls you to edit `train.py`, then runs it, parses the metric, updates `progress.csv`, commits, and repeats. **You are only responsible for editing `train.py`** (and installing packages in the shell). Do NOT run `train.py`, do NOT write to `progress.csv`, do NOT run `git commit` or `git push`.

Up to {max_experiments} experiments (0-indexed). The runner stops early based on your DESCRIPTION or if you return STOP.

---

### Your task each experiment

**Experiment 0 (implement baseline):**
The provided `train.py` is a **skeleton** — it only loads and splits the data, no model.
1. Read `train.py` to see the variable names already in scope (`X_train/y_train`, `train_seqs/train_labels`, `train_ds/val_ds`, etc.)
2. Reason about the best model for this domain, data size, and compute budget using the Model Selection principles above
3. Implement the complete training script — model, training loop, validation, metric evaluation
4. Install any needed packages in the shell first (`pip install <pkg>`), not inside `train.py`
5. Final line of train.py must print: `BEST_VAL_{metric_name.upper()}: {{value:.6f}}`

**Experiments 1 and later:**
Edit `train.py` to make a change most likely to improve `{metric_name}`. Study `progress.csv` first — do not repeat a failed change. If the last 3 experiments all failed, make a bolder change (different model family, different training strategy).

---

## progress.csv
Read-only reference for you. The Python runner writes one row per experiment automatically — never write to it yourself.
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
