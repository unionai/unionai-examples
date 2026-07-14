"""
ResearchAgent — Python-managed training loop with claude-agent-sdk for code proposals.

Python runs train.py directly, parses metrics, writes progress.csv, and commits.
claude-agent-sdk (ClaudeAgentOptions / query()) is called once per experiment to
read train.py and apply ONE focused change (max_turns=10, acceptEdits mode).

This separates "deciding what to change" (Claude's job, fast, bounded)
from "running the training" (Python's job, with an explicit timeout).

Every step of the loop (CLI setup, git clone, baseline implementation, change
proposals, training runs, crash fixes, commits, PR creation, convergence
analysis) is decorated with @flyte.trace (sync functions are supported from
flyte 2.5) so it appears as a traced action in the Flyte UI with inputs,
outputs, and timing — all inside the single run_research task container.

These are traces rather than separate @research_env.task's because every step
depends on container-local state (the repo clone in /tmp, the installed
Node/claude CLI, the downloaded dataset, the warm GPU): a child task would
start in a fresh container and lose all of it.

Two things to keep in mind when editing the traced functions:
- A trace's identity is (function name + inputs hash) — a repeated call with
  identical inputs replays the recorded result instead of re-executing.
  Per-experiment steps therefore take exp_id (and run_training an attempt
  counter) so every call is a distinct traced action.
- Tracing needs the Flyte task context, which lives in contextvars. The
  run_research task is a sync task, and Flyte runs sync task bodies in a
  dedicated thread with the task's contextvars copied in — so the traced
  calls below see the task context without any extra plumbing.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import flyte

GITHUB_USERNAME = "parnianz"
GITHUB_EMAIL    = "parnianzargham@gmail.com"


# ---------------------------------------------------------------------------
# Node.js + Claude Code CLI install
# (claude-agent-sdk is a pip package but needs the claude CLI binary at runtime)
# ---------------------------------------------------------------------------

@flyte.trace
def install_claude_cli() -> str:
    """Download Node.js and install the Claude Code CLI needed by claude-agent-sdk;
    returns the CLI version."""
    subprocess.run(["apt-get", "update", "-y"], check=False, capture_output=True)

    node_url = "https://nodejs.org/dist/v20.19.0/node-v20.19.0-linux-x64.tar.gz"
    node_tar = Path("/tmp/node.tar.gz")
    print("Downloading Node.js...", flush=True)
    urllib.request.urlretrieve(node_url, node_tar)
    size_mb = node_tar.stat().st_size / 1024 / 1024
    print(f"  {size_mb:.1f} MB downloaded", flush=True)
    if size_mb < 1:
        raise RuntimeError(f"Node.js download appears corrupt ({size_mb:.2f} MB) — network restricted?")

    node_dir = Path("/tmp/node")
    node_dir.mkdir(exist_ok=True)
    with tarfile.open(node_tar, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.split("/", 1)[-1]]
        for m in members:
            m.name = m.name.split("/", 1)[-1]
        tar.extractall(str(node_dir), members=[m for m in members if m.name])

    node_bin = str(node_dir / "bin")
    os.environ["PATH"] = node_bin + ":" + os.environ.get("PATH", "")

    npm_prefix = "/tmp/npm-global"
    Path(npm_prefix).mkdir(exist_ok=True)
    subprocess.run(
        ["npm", "install", "-g", "--prefix", npm_prefix, "@anthropic-ai/claude-code"],
        check=True, capture_output=True,
    )
    os.environ["PATH"] = str(Path(npm_prefix) / "bin") + ":" + os.environ["PATH"]
    ver = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"Claude Code CLI ready: {ver}", flush=True)
    return ver


# ---------------------------------------------------------------------------
# Helper: run async coroutine from a sync context without fighting Flyte's loop
# ---------------------------------------------------------------------------

def _run_in_new_loop(coro) -> None:
    """Run an async coroutine in a dedicated thread with a fresh event loop."""
    errors: list[Exception] = []

    def _thread():
        loop = asyncio.new_event_loop()
        # Do NOT call asyncio.set_event_loop() — it can corrupt thread-local loop
        # state on Python 3.12+ and is not needed since we call loop methods directly.
        try:
            loop.run_until_complete(coro)
        except Exception as exc:
            errors.append(exc)
        finally:
            # Cancel any tasks the SDK left open before closing the loop.
            # Without this, loop.close() leaves dangling futures that can leak
            # into the outer asyncio runner and cause cleanup errors.
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=_thread, daemon=False)
    t.start()
    t.join()
    if errors:
        raise errors[0]


# ---------------------------------------------------------------------------
# Claude Agent SDK — baseline implementation + improvement proposals
# ---------------------------------------------------------------------------

@flyte.trace
def implement_baseline(exp_dir: str, metric_name: str, model: str) -> str:
    """
    Experiment 0: agent reads the data-loading skeleton + program.md,
    chooses a model, installs packages, and writes the complete train.py.
    Returns a description of what was implemented.
    """
    prompt = f"""You are implementing the baseline training script for an AutoML experiment.

`train.py` is a skeleton — it loads and splits the data but has no model, training loop, or metric evaluation.

Your job:
1. Read `train.py` to understand the data variables already in scope (e.g. X_train/y_train for tabular, train_seqs/train_labels for sequence, dataset/train_idx/val_idx/native_h/native_w/img_channels for image)
2. Read `program.md` for dataset details (domain, task, samples, class distribution), the Model Selection principles, and hard constraints
3. Choose the best approach using this decision process (read program.md for N, F, numeric/categorical counts, L, and domain):
   - **Tabular (non-temporal)** — ladder by N and feature types:
     a. N < 10,000 → LightGBM/XGBoost/CatBoost. Use 5-fold stratified CV for reliability. Don't try deep learning yet.
     b. 10,000 ≤ N < 100,000 → GBM first; try small MLP (2–3 layers) only if GBM plateaus.
     c. N ≥ 100,000 → GBM still competitive; TabNet or FT-Transformer become viable — try both.
     d. Heavy categorical (many columns, high cardinality >20 unique) → CatBoost or MLP with entity embeddings.
     e. Mostly numeric → LightGBM/XGBoost with one-hot/ordinal encoding.
     Feature engineering before switching models: log/sqrt for skewed numerics, target-encode high-cardinality cats, add missingness indicator flags, drop zero-importance features after first GBM fit.
   - **Time series** (domain contains ecg/eeg/sensor/temporal/timeseries/signal/etc.) — ladder by N and L:
     a. N < 1,000 → Classical ML: rolling stats + FFT features → XGBoost/Random Forest
     b. N ≥ 1,000, L ≤ 200 → 1D-CNN (3–4 conv blocks + global avg pool + FC). Reshape: (N, C, L).
     c. N ≥ 1,000, 200 < L ≤ 1,000, N < 50k → 1D-CNN + optional LSTM head
     d. N ≥ 1,000, 200 < L ≤ 1,000, N ≥ 50k → Transformer with patching (PatchTST-style)
     e. N ≥ 1,000, L > 1,000, N ≥ 5k → Bidirectional LSTM/GRU; N < 5k → patch then CNN
   - **Sequence (biological — DNA/RNA/protein)** (work through this ladder, do NOT jump straight to a transformer):
     a. Fixed-length short sequences (≤200 chars): split each char position into a column → LightGBM
     b. Any sequence: k-mer TF-IDF (ngram_range=(3,6)) → LightGBM or logistic regression
     c. If a and b plateau: 1D CNN on one-hot encoded sequences
     d. If CNN plateaus AND N≥5k: frozen domain-specific transformer + linear probe
     e. If frozen probe plateaus AND N≥5k: two-phase fine-tuning (Phase 1 frozen cache, Phase 2 backbone.train())
   - **Image** — variables in scope from skeleton: `dataset` (ImageFolder), `train_idx`, `val_idx`, `native_h`, `native_w`, `img_channels`, `num_classes`, `train_transform`, `val_transform`. Build DataLoaders with `SubsetRandomSampler(train_idx/val_idx)` — do NOT re-split. Scale-based strategy:
     - N < 1,000: freeze backbone, extract embeddings once (FP16, no_grad, batch=256), cache to disk, fit sklearn LogisticRegression/SVC on cached embeddings
     - 1k ≤ N < 10k: lightweight ImageNet backbone (EfficientNet-B0, MobileNetV3-Small, ResNet18) from `timm`. Phase 1: frozen backbone + linear head (LR=1e-3, 10–15 epochs). Phase 2: unfreeze all, backbone LR=1e-4 / head LR=1e-3, cosine schedule.
     - 10k ≤ N < 50k: lightweight-to-medium backbone (EfficientNet-B0/B2, ResNet34), full fine-tuning, aggressive augmentation (RandomCrop, ColorJitter, RandomErasing).
     - N ≥ 50k: heavier backbone (EfficientNet-B4, ResNet50), full fine-tuning with torch.cuda.amp for larger batches.
     Always use `native_h` and `native_w` from the skeleton in ALL transforms — never hardcode resolution. For `img_channels==1` add Grayscale transform. Normalize with ImageNet mean/std for ImageNet-pretrained models.
4. Install any packages needed: `pip install <pkg>` in the shell (NEVER inside train.py)
5. Implement the complete training script in train.py: model loading, training loop, per-epoch validation, metric evaluation. This is experiment 0 — make it a solid, runnable baseline
6. The final printed line of train.py must be exactly: BEST_VAL_{metric_name.upper()}: <value:.6f>

HARD CONSTRAINTS:
- DATA_PATH env var is always set — use it directly. NEVER hardcode any path, NEVER add a try/except fallback to a different hardcoded path.
- No pip install inside train.py — install in the shell first.
- num_workers=0 in all DataLoaders.
- No threshold calibration on the validation set — do NOT use scipy.optimize, differential_evolution, or any search over val labels to pick per-class thresholds. This leaks val labels and inflates the reported metric. Use argmax/sigmoid outputs directly.
- No double normalization — apply exactly one normalization pass (either per-sample OR dataset-level, not both).

DO NOT do any of the following — the Python runner handles them automatically:
- Do NOT run train.py (do not execute `python train.py`)
- Do NOT write to progress.csv
- Do NOT run git commit or git push
Your only job is to write train.py and install any needed packages.

REQUIRED: After all tool calls are done, your final text response MUST contain this line (not in a shell command, in your reply text):
DESCRIPTION: baseline: <one-line description of model chosen and why>
"""
    collected: list[str] = []

    async def _run() -> None:
        from claude_agent_sdk import query, ClaudeAgentOptions
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage
        async for msg in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                permission_mode="acceptEdits",
                allowed_tools=["Read", "Edit", "Write", "Bash"],
                max_turns=40,
                model=model,
                cwd=exp_dir,
                env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
            ),
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        collected.append(block.text)
                        print(block.text, flush=True)

    try:
        _run_in_new_loop(_run())
    except Exception as e:
        print(f"implement_baseline failed: {e}", flush=True)
        return "baseline: sdk_error"

    full_text = "\n".join(collected)
    match = re.search(r"DESCRIPTION:\s*(.+)", full_text)
    return match.group(1).strip() if match else "baseline"


@flyte.trace
def propose_change(
    exp_dir: str,
    metric_name: str,
    higher: bool,
    exp_id: int,
    max_experiments: int,
    time_budget_s: int,
    model: str,
) -> tuple[bool, str]:
    """
    Launch a short Claude Code agent session (max_turns=10) in exp_dir.
    Claude reads train.py + progress.csv and applies ONE focused edit.
    Returns (should_stop, description).
    """
    history = _read_history(Path(exp_dir) / "progress.csv")
    history_text = "\n".join(
        f"  exp {r['experiment_id']}: {r['description']} → "
        f"{r.get('metric_name', metric_name)}={r.get('metric_value', 'n/a')} "
        f"improved={r.get('improved', 'false')}"
        for r in history
    ) or "  (none yet)"

    prompt = f"""You are an AI researcher. Your goal: {'maximize' if higher else 'minimize'} `{metric_name}`.

Experiment history ({exp_id} of {max_experiments} used):
{history_text}

Everything in train.py is fair game — model architecture, model family, optimizer, loss, hyperparameters, training strategy, data augmentation, feature engineering. Read program.md for dataset details and model selection principles.

Steps:
1. Read train.py and progress.csv (and program.md if you need dataset/domain context)
2. If the last run crashed or produced no metric: fix the error first, then improve
3. Diagnose why performance is where it is — reason about model capacity, training dynamics, and data characteristics
4. Decide what to change. If the current model family has hit its ceiling, switch to a completely different one. If the approach is fundamentally wrong, replace it entirely. Install any new packages needed: `pip install <pkg>` in the shell (never inside train.py)
5. Edit train.py with your changes
6. REQUIRED: After all tool calls are done, your final text response MUST contain this line (not via Bash, in your reply text):
   DESCRIPTION: <what you changed and why, one concise line>

DO NOT do any of the following — the Python runner handles them automatically:
- Do NOT run train.py (do not execute `python train.py`)
- Do NOT write to or modify progress.csv
- Do NOT run git commit or git push
Your only job is to edit train.py and install any needed packages.

HARD CONSTRAINTS (never override):
- num_workers=0 in all DataLoaders
- No pip install inside train.py
- DATA_PATH env var is always set by the runner — use it directly, never hardcode any path, never add a try/except fallback to a hardcoded path (e.g. NO: `except: pd.read_csv('/tmp/something/old.csv')`)
- Never pass raw sequences/strings to sklearn or numpy — always encode first
- Time budget: {time_budget_s}s on a T4 GPU (16 GB VRAM) — estimate cost before choosing a model
- Memory: 32 GB RAM, 16 GB VRAM — if OOMKilled (exit 137), reduce batch size or model size
- NEVER use ignore_mismatched_sizes=True — silently destroys pretrained weights; fix the config/architecture instead
- Always pass config=config to AutoModel.from_pretrained
- If AutoModel raises size-mismatch: use task-specific class (AutoModelForMaskedLM etc.) and extract encoder sub-module (.bert/.roberta/.esm/.distilbert)
- Focal loss alpha = inverse class frequency (1/count, normalized) — never raw frequencies
- No threshold calibration on the validation set — do NOT use scipy.optimize, differential_evolution, or any fitting over val labels to pick per-class thresholds. This leaks val labels and inflates the reported metric. Use argmax/sigmoid outputs directly; if calibration is needed use a separate held-out calibration split.
- No double normalization — apply exactly one normalization pass (per-sample OR dataset-level, not both in sequence).
- CatBoost train_dir: always pass `train_dir='/tmp/catboost_info'` when constructing any CatBoost model — without this it writes a catboost_info/ folder into the experiment directory which gets committed to git.

DIAGNOSIS GUIDE:
- No metric output / script ran but printed nothing → train.py is a skeleton with no model; implement the full training script
- Metric near zero → model or feature extraction is wrong; replace the approach
- Tabular task stuck after GBM baseline:
  (a) Try feature engineering first: log transforms for skewed columns, interaction terms, target encoding for high-cardinality categoricals, missingness flags
  (b) If still stuck: try a different GBM (CatBoost if heavy categoricals, XGBoost if LightGBM plateau)
  (c) If GBM family has plateaued and N≥10k: try small MLP (2–3 layers, batch norm, dropout)
  (d) If N≥100k and MLP plateaus: try TabNet or FT-Transformer
  (e) Check class imbalance — if largest class >2× smallest, add class_weight='balanced' or focal loss
- Time series task (ecg/eeg/sensor/temporal domain) — check N and L from program.md, then follow this ladder:
  (a) N < 1,000 → Classical ML: rolling stats + FFT → XGBoost. Deep nets overfit here.
  (b) N ≥ 1,000, L ≤ 200 → 1D-CNN; reshape X to (N, C, L) before Conv1d. Start here for ECG/EEG short windows.
  (c) N ≥ 1,000, L 200–1,000, N < 50k → 1D-CNN + LSTM head
  (d) N ≥ 1,000, L 200–1,000, N ≥ 50k → Transformer with patching
  (e) L > 1,000, N ≥ 5k → BiLSTM/GRU; N < 5k → patch then CNN
  Multivariate (C > 1): prefer CNN/Transformer over LSTM. Streaming: always use LSTM/GRU.
- Biological sequence task — do NOT jump straight to a transformer. Work through this ladder:
  (a) Fixed-length short sequences (≤200 chars): positional tabular features + LightGBM — often strongest
  (b) k-mer TF-IDF (ngram_range=(3,6)) + LightGBM or logistic regression — fast, no GPU
  (c) 1D CNN on one-hot — captures local motifs
  (d) Frozen domain transformer + linear probe — only if (a)–(c) plateau and N≥5k
  (e) Two-phase fine-tuning — only if frozen probe plateaus and N≥5k
- Sequence/time-series metric stuck with frozen backbone → ceiling reached; move to Phase 2 (backbone.train().float(), new DataLoader, backbone LR=1e-5 / head LR=1e-4)
- Image metric stuck at frozen-backbone ceiling → unfreeze and fine-tune: `backbone.train()`, add backbone param group with LR=1e-4 (10× lower than head LR), cosine schedule
- Image OOM → reduce batch size first; if still OOM enable mixed precision (`torch.cuda.amp.autocast + GradScaler`) before reducing model size
- Image metric improving but slowly → check augmentation strength; if val loss is oscillating, reduce LR; if train/val gap is large (overfitting), add dropout, weight decay, or stronger augmentation (RandomErasing, Cutmix)
- Image with class imbalance → replace uniform DataLoader with `WeightedRandomSampler(weights=inverse_class_freq, num_samples=len(train_idx))` — simpler and more reliable than SMOTE for images
- Metric improving slowly → try LR schedule, larger batch, or longer training
- Train metric good, val metric poor → overfitting; add regularization or reduce model size
- OOM → reduce batch size, then sequence/window length, then model size

If further improvement is not possible, your final text response must be exactly the single word `STOP` on its own line (and nothing else — no DESCRIPTION).
"""

    collected: list[str] = []

    async def _run_agent() -> None:
        from claude_agent_sdk import query, ClaudeAgentOptions
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

        async for msg in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                # acceptEdits auto-approves file edits without --dangerously-skip-permissions
                # so it works as root inside the container.
                permission_mode="acceptEdits",
                allowed_tools=["Read", "Edit", "Write", "Bash"],
                max_turns=30,
                model=model,
                cwd=exp_dir,
                env={"ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "")},
            ),
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        collected.append(block.text)
                        print(block.text, flush=True)
            elif isinstance(msg, ResultMessage) and msg.is_error:
                print(f"Agent SDK error: {msg.result}", flush=True)

    try:
        _run_in_new_loop(_run_agent())
    except Exception as e:
        print(f"propose_change failed: {e} — keeping current train.py", flush=True)
        return False, f"sdk_error_{exp_id}"

    full_text = "\n".join(collected)
    if re.search(r"^\s*STOP\s*$", full_text, re.MULTILINE):
        return True, ""

    match = re.search(r"DESCRIPTION:\s*(.+)", full_text)
    description = (match.group(1).strip() if match else f"change_{exp_id}")
    return False, description


@flyte.trace
def analyze_convergence(
    history: str,
    program_text: str,
    best_str: str,
    metric_name: str,
    higher: bool,
    num_experiments: int,
    model: str,
) -> dict:
    """
    Ask Claude to judge whether training converged to a satisfactory result.
    Returns a dict: {is_good, summary, reasons, suggestions}
    """
    import anthropic

    prompt = f"""You are an ML expert reviewing the results of an automated training loop.

## Task
- Metric: {metric_name} ({'higher is better' if higher else 'lower is better'})
- Best value achieved: {best_str}
- Experiments run: {num_experiments}

## Dataset & Goal (from program.md)
{program_text}

## Experiment history (progress.csv)
{history}

## Instructions
Decide whether the best {metric_name} of {best_str} is a satisfactory result given the dataset (size, class distribution, modality, task difficulty).

If the result IS satisfactory: set is_good=true and write a short positive summary.

If the result is NOT satisfactory (poor metric, no improvement across experiments, all crashes, stuck model): set is_good=false and explain specifically what went wrong — look at the experiment trajectory, what was tried, what failed. Be concrete: reference actual values from the history.

Return ONLY a JSON object with exactly these fields:
{{
  "is_good": true or false,
  "summary": "one concise sentence verdict",
  "reasons": ["specific reason 1", "specific reason 2"],
  "suggestions": ["actionable suggestion 1", "actionable suggestion 2"]
}}
reasons and suggestions can be empty lists if is_good is true.
Return ONLY the JSON, no extra text."""

    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.rstrip())
        return json.loads(raw)
    except Exception as e:
        print(f"analyze_convergence failed: {e}", flush=True)
        return {
            "is_good": not best_str.startswith("N/A"),
            "summary": f"Best {metric_name}: {best_str}",
            "reasons": [],
            "suggestions": [],
        }


@flyte.trace
def fix_crash(exp_id: int, train_py: str, error_output: str, model: str) -> str | None:
    """Use Anthropic Python SDK to fix a crashed train.py.
    exp_id is an input only so each experiment's fix is a distinct traced action."""
    import anthropic
    client = anthropic.Anthropic()
    prompt = f"""The training script crashed. Fix it so it trains successfully and prints the final metric.

ERROR (last 1000 chars):
{error_output[-1000:]}

train.py:
```python
{train_py}
```

Fix the crash using these principles:

**Model loading errors** (size mismatch, missing attributes, ImportError):
- NEVER add ignore_mismatched_sizes=True — it silently randomly-initializes layers and destroys pretrained weights
- Always pass config=config to from_pretrained — without it, object.__setattr__ config patches are ignored
- If AutoModel raises a size-mismatch (checkpoint has extra MLM/causal head): switch to the task-specific class (AutoModelForMaskedLM, AutoModelForCausalLM, etc.) and extract the encoder sub-module (.bert, .roberta, .esm, .distilbert, etc.)
- If a transformers API function is missing (ImportError): it was likely moved between versions — try importing from pytorch_utils, modeling_utils, or the model-specific module
- If a config attribute is missing (AttributeError on config.xxx): use object.__setattr__(config, 'xxx', default_value) BEFORE calling from_pretrained, and pass config=config

**Model selection** (if the model itself is the problem):
- Switch to a different model of similar capability that avoids the compatibility issue
- Stay under 200M parameters and 200MB download
- Avoid models requiring flash-attention

**Other errors**: fix the root cause directly — do not add broad try/except to suppress errors.

Respond with ONLY the fixed train.py (no markdown fences, no explanation).
"""
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        fixed = resp.content[0].text.strip()
        fixed = re.sub(r"^```\w*\n?", "", fixed)
        fixed = re.sub(r"\n?```$", "", fixed.rstrip())
        return fixed
    except Exception as e:
        print(f"fix_crash error: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

@flyte.trace
def clone_experiment_branch(github_repo: str, branch_name: str) -> str:
    """Clone the branch DesignAgent pushed; returns the local clone path.
    The GitHub token is read from the environment so it is not recorded as a
    trace input."""
    github_token = os.environ.get("GITHUB_TOKEN", "")
    clone_dir = Path(f"/tmp/automl_research_{branch_name}")
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    repo_url = f"https://x-access-token:{github_token}@github.com/{github_repo}.git"
    subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)
    subprocess.run(["git", "checkout", branch_name], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.email", GITHUB_EMAIL], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.name",  GITHUB_USERNAME], cwd=clone_dir, check=True)
    return str(clone_dir)


@flyte.trace
def commit_and_push(
    exp_id: int,
    description: str,
    metric_name: str,
    metric_value: float | None,
    clone_dir: str,
) -> str:
    """Commit and push an improved train.py; returns the short commit hash."""
    val_str = f"{metric_value:.4f}" if metric_value is not None else "n/a"
    msg = f"exp{exp_id}: {description} | {metric_name}={val_str}"
    subprocess.run(["git", "add", "-A"], cwd=clone_dir, check=False)
    subprocess.run(["git", "commit", "-m", msg], cwd=clone_dir, check=False)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, check=False)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=clone_dir, capture_output=True, text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# PR
# ---------------------------------------------------------------------------

@flyte.trace
def create_pull_request(
    github_repo: str,
    branch_name: str,
    metric_name: str,
    best_value: str,
) -> str:
    """Open (or find) the results PR; returns its URL."""
    github_token = os.environ.get("GITHUB_TOKEN", "")
    try:
        from github import Auth, Github, GithubException
        auth = Auth.Token(github_token)
        gh   = Github(auth=auth)
        repo = gh.get_repo(github_repo)

        default_branch = repo.default_branch
        if default_branch == branch_name or default_branch.startswith("automl-"):
            # If the repo was empty when the first experiment branch was
            # pushed, GitHub made that branch the default. PRs must not
            # target an experiment branch — ensure a main branch exists
            # (at this branch's root commit) and use it as the base.
            base = "main"
            try:
                repo.get_branch(base)
            except GithubException:
                root_sha = list(repo.get_commits(sha=branch_name))[-1].sha
                repo.create_git_ref(ref=f"refs/heads/{base}", sha=root_sha)
            try:
                # Needs the Administration permission — best effort only.
                repo.edit(default_branch=base)
            except GithubException:
                print(f"Could not change repo default branch (token lacks admin) — set it to {base} manually", flush=True)
            print(f"Repo default branch is {default_branch}; using {base} as PR base", flush=True)
            default_branch = base

        existing = list(repo.get_pulls(
            state="open",
            head=f"{github_repo.split('/')[0]}:{branch_name}",
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
- `program.md` — research instructions

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
        return f"https://github.com/{github_repo}/tree/{branch_name}"


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

@flyte.trace
def run_training(
    exp_id: int,
    attempt: int,
    exp_dir: str,
    local_data_path: str,
    timeout: int = 3600,
) -> tuple[str, bool]:
    """Run train.py as a subprocess with a hard timeout. Streams output in real time.
    exp_id and attempt are inputs only so each run is a distinct traced action —
    a trace with identical inputs would replay the recorded result instead of
    re-running the training."""
    env = {
        **os.environ,
        "DATA_PATH": local_data_path,
        "PYTHONUNBUFFERED": "1",
        # Ensure HuggingFace downloads are always allowed and cached to a writable path.
        # HF_HUB_OFFLINE=1 (sometimes set in container environments) silently blocks all downloads.
        "HF_HUB_OFFLINE": "0",
        "HF_HOME": "/tmp/hf_model_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_model_cache",
        "HF_HUB_CACHE": "/tmp/hf_model_cache",
    }

    hf_diag = {k: env.get(k, "<not set>") for k in (
        "HF_HUB_OFFLINE", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE",
        "HF_HUB_DISABLE_PROGRESS_BARS", "HUGGING_FACE_HUB_TOKEN", "http_proxy", "https_proxy",
    )}
    print(f"[HF ENV] {hf_diag}", flush=True)

    proc = subprocess.Popen(
        ["python", "train.py"],
        cwd=exp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    timed_out = False

    def _kill_after():
        nonlocal timed_out
        time.sleep(timeout)
        if proc.poll() is None:
            timed_out = True
            proc.kill()

    threading.Thread(target=_kill_after, daemon=True).start()

    lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        lines.append(line)

    proc.wait()

    output = "\n".join(lines)
    if timed_out:
        output += f"\n[TIMEOUT: training killed after {timeout}s]"
    return output, proc.returncode != 0 or timed_out


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    MODEL = "claude-sonnet-4-6"

    def __init__(self, github_repo: str):
        self.github_repo = github_repo

    def run(
        self,
        branch_name: str,
        experiment_folder: str,
        local_data_path: str,
        max_experiments: int = 20,
        time_budget_per_experiment_seconds: float = 3600.0,
    ) -> str:
        # 1. Install Node.js + claude CLI (needed by claude-agent-sdk at runtime)
        install_claude_cli()

        # 2. Clone the branch DesignAgent pushed
        clone_dir = Path(clone_experiment_branch(self.github_repo, branch_name))
        exp_dir   = clone_dir / experiment_folder

        # 3. Make data readable
        subprocess.run(["chmod", "-R", "a+rX", local_data_path], check=False)

        # 4. Metric direction from program.md
        metric_name, higher = _parse_metric_direction(exp_dir / "program.md")

        # 5. Init progress.csv
        progress_csv = exp_dir / "progress.csv"
        if not progress_csv.exists():
            progress_csv.write_text(
                "experiment_id,description,model_name,metric_name,metric_value,improved,duration_s,commit,notes\n"
            )

        timeout = int(time_budget_per_experiment_seconds)
        best_metric: float | None = None
        best_train_py = (exp_dir / "train.py").read_text()

        for exp_id in range(max_experiments):
            print(f"\n{'='*60}", flush=True)
            print(f">>> [AUTOTRAIN] Starting experiment {exp_id}", flush=True)

            if exp_id == 0:
                # Implement the baseline: agent reads skeleton + program.md,
                # chooses a model, installs packages, writes complete train.py
                description = implement_baseline(str(exp_dir), metric_name, self.MODEL)
            else:
                should_stop, description = propose_change(
                    exp_dir=str(exp_dir),
                    metric_name=metric_name,
                    higher=higher,
                    exp_id=exp_id,
                    max_experiments=max_experiments,
                    time_budget_s=timeout,
                    model=self.MODEL,
                )
                if should_stop:
                    print(
                        f"Claude decided no further improvement possible — stopping at exp {exp_id}.",
                        flush=True,
                    )
                    break

            # Run training (Python subprocess, hard timeout)
            t0 = time.time()
            output, crashed = run_training(exp_id, 0, str(exp_dir), local_data_path, timeout)
            duration = time.time() - t0

            metric_value = _parse_metric_from_output(output, metric_name)

            # If crashed (any experiment), attempt one auto-fix via simple LLM call
            if crashed:
                print(f"Exp {exp_id} crashed — attempting auto-fix...", flush=True)
                fixed = fix_crash(
                    exp_id=exp_id,
                    train_py=(exp_dir / "train.py").read_text(),
                    error_output=output,
                    model=self.MODEL,
                )
                if fixed:
                    (exp_dir / "train.py").write_text(fixed)
                    best_train_py = fixed
                    t0 = time.time()
                    output, crashed = run_training(exp_id, 1, str(exp_dir), local_data_path, timeout)
                    duration = time.time() - t0
                    metric_value = _parse_metric_from_output(output, metric_name)

            print(f">>> [AUTOTRAIN] Result: {metric_name}={metric_value}", flush=True)

            # Determine improvement
            improved = False
            if metric_value is not None and not crashed:
                if best_metric is None:
                    improved = True
                    best_metric = metric_value
                    best_train_py = (exp_dir / "train.py").read_text()
                elif (higher and metric_value > best_metric) or (not higher and metric_value < best_metric):
                    improved = True
                    best_metric = metric_value
                    best_train_py = (exp_dir / "train.py").read_text()

            # Commit improved, revert otherwise
            commit_hash = ""
            if improved:
                (exp_dir / "best_train.py").write_text(best_train_py)
                commit_hash = commit_and_push(
                    exp_id=exp_id,
                    description=description,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    clone_dir=str(clone_dir),
                )
            else:
                (exp_dir / "train.py").write_text(best_train_py)

            _append_csv(
                progress_csv,
                exp_id=exp_id,
                description=description,
                metric_name=metric_name,
                metric_value=metric_value,
                improved=improved,
                duration_s=duration,
                commit=commit_hash,
                notes=_crash_note(output) if crashed else "",
            )
            print(f">>> [AUTOTRAIN] CSV updated", flush=True)

        # Final commit — push full progress.csv regardless of which experiments improved
        subprocess.run(["git", "add", str(progress_csv)], cwd=clone_dir, check=False)
        subprocess.run(
            ["git", "commit", "-m", "autotrain: final progress.csv"],
            cwd=clone_dir, check=False,
        )
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=clone_dir, check=False)

        # Post-loop diagnostics
        git_log = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            cwd=clone_dir, capture_output=True, text=True,
        )
        print(f"\nGit log after training loop:\n{git_log.stdout}", flush=True)
        print(f"\nprogress.csv:\n{progress_csv.read_text()}", flush=True)

        best_value = f"{best_metric:.6f}" if best_metric is not None else "n/a"
        pr_url     = create_pull_request(self.github_repo, branch_name, metric_name, best_value)

        program_md = exp_dir / "program.md"
        convergence = analyze_convergence(
            history=progress_csv.read_text() if progress_csv.exists() else "No history available.",
            program_text=program_md.read_text()[:1500] if program_md.exists() else "",
            best_str=f"{best_metric:.6f}" if best_metric is not None else "N/A (all experiments crashed)",
            metric_name=metric_name,
            higher=higher,
            num_experiments=exp_id + 1,
            model=self.MODEL,
        )

        return (
            f"AutoTrain complete\n"
            f"  Branch : {branch_name}\n"
            f"  Best {metric_name}: {best_value}\n"
            f"  PR     : {pr_url}\n"
            f"\n---CONVERGENCE_JSON---\n"
            f"{json.dumps(convergence)}"
        )


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def _parse_metric_from_output(output: str, metric_name: str) -> float | None:
    """
    Parse final metric value from stdout.
    Expects BEST_VAL_{METRIC}: <value> (train.py convention), falls back to
    looser {metric}: <value> pattern.
    """
    patterns = [
        rf"BEST_VAL_{re.escape(metric_name.upper())}:\s*([\d.eE+\-]+)",
        rf"BEST_VAL_{re.escape(metric_name.upper())}=([\d.eE+\-]+)",
        rf"\b{re.escape(metric_name)}\s*[=:]\s*([\d.eE+\-]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, output, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _crash_note(output: str) -> str:
    """Summarize what happened: timeout message or exception type + message."""
    if "[TIMEOUT:" in output:
        for line in reversed(output.splitlines()):
            if "[TIMEOUT:" in line:
                return line.strip()
    lines = [l.strip() for l in output.splitlines() if l.strip()]
    error_lines = [l for l in lines if any(kw in l.lower() for kw in ("error", "exception"))]
    if error_lines:
        return error_lines[-1]
    return lines[-1] if lines else "crashed"


def _append_csv(
    path: Path,
    exp_id: int,
    description: str,
    metric_name: str,
    metric_value: float | None,
    improved: bool,
    duration_s: float,
    commit: str,
    notes: str,
) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_id,
            description,
            "",
            metric_name,
            f"{metric_value:.6f}" if metric_value is not None else "",
            str(improved).lower(),
            f"{duration_s:.1f}",
            commit,
            notes,
        ])


def _read_history(progress_csv: Path) -> list[dict]:
    if not progress_csv.exists():
        return []
    try:
        with open(progress_csv) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Metric direction (read from program.md)
# ---------------------------------------------------------------------------

def _parse_metric_direction(program_md: Path) -> tuple[str, bool]:
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
