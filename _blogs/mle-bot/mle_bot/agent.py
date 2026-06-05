"""MLE Agent — orchestrates ML experiments using Flyte's durable sandbox.

The agent:
  1. Profiles the dataset using a trusted tool (data never touches the LLM).
  2. Asks OpenAI to design a set of experiments (algorithms, hyperparams, feature strategy).
  3. For each experiment, generates Monty orchestration code and executes it via
     flyte.sandbox.orchestrate_local(), which dispatches the heavy compute as durable tasks.
  4. Analyzes results, iterates if needed.
  5. Produces a model card summarizing the winning model.

The Monty sandbox ensures the LLM-generated orchestration code is safe — it can only
call the pre-approved tool functions and has no access to imports, network, or filesystem.
"""

import asyncio
import inspect
import json
import os
import textwrap
from dataclasses import dataclass

import flyte
import flyte.sandbox
from flyte.io import File

from mle_bot.schemas import ExperimentConfig, InitialDesign, IterationDecision

from mle_bot.environments import agent_env
from mle_bot.tools.data import profile_dataset, split_dataset
from mle_bot.tools.evaluation import evaluate_model, rank_experiments
from mle_bot.tools.exploration import explore_dataset
from mle_bot.tools.features import engineer_features
from mle_bot.tools.predictions import get_predictions
from mle_bot.tools.resampling import resample_dataset
from mle_bot.tools.selection import select_features
from mle_bot.tools.training import train_model

# All tools exposed to the Monty sandbox.
# Keys must match the function names used in LLM-generated orchestration code.
TOOLS = [
    profile_dataset, split_dataset, explore_dataset,
    engineer_features, resample_dataset, select_features,
    train_model, get_predictions, evaluate_model, rank_experiments,
]
TOOLS_BY_NAME = {t.func.__name__ if hasattr(t, "func") else t.__name__: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _tool_signatures() -> str:
    """Build a summary of available tool signatures and docstrings for the system prompt."""
    parts = []
    for t in TOOLS:
        func = t.func if hasattr(t, "func") else t
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        # Trim docstring to first 40 lines so prompt stays manageable
        doc_lines = doc.splitlines()[:40]
        short_doc = "\n    ".join(doc_lines)
        parts.append(f"async def {func.__name__}{sig}:\n    \"\"\"{short_doc}\"\"\"\n    ...")
    return "\n\n".join(parts)


def _build_orchestration_system_prompt(profile: dict) -> str:
    monty_rules = flyte.sandbox.ORCHESTRATOR_SYNTAX_PROMPT
    tools_section = _tool_signatures()
    is_imbalanced = profile.get("is_imbalanced", False)
    class_balance = profile.get("class_balance", {})
    columns = profile.get("columns", [])
    numeric_cols = profile.get("numeric_columns", [])
    categorical_cols = profile.get("categorical_columns", [])
    corr = profile.get("feature_target_corr", {})
    corr_str = ", ".join(f"{k}: {v:+.3f}" for k, v in list(corr.items())[:8]) if corr else "n/a"
    shape = profile.get("shape", [0, 0])
    return f"""\
You are an expert ML engineer. Your job is to design and write the best possible
pipeline for a machine learning experiment, then generate the Python orchestration
code to execute it.

The code runs inside a restricted Monty sandbox. The last expression in your code
is returned as the result. All tool calls are made like regular function calls —
you do NOT need to await them.

## Dataset context

Shape: {shape[0]:,} rows × {shape[1]} columns
Numeric features: {numeric_cols}
Categorical features (excluded from model — not supported): {categorical_cols}
Class balance: {class_balance} — imbalanced: {is_imbalanced}
Feature-target correlations (raw, point-biserial): {corr_str}

## General ML best practices — apply these based on the dataset context above

**Feature engineering** (engineer_features tool):
- Sequential/time-series data (timestamp column present, rows ordered over time):
  rolling window features (means, stds, min/max) capture trends that point-in-time
  readings miss. Lag features capture recent history. Choose window sizes relative
  to the prediction horizon and temporal resolution of the data.
- Tabular cross-sectional data: normalization helps linear models and distance-based
  methods. Interaction terms can help if correlations are weak individually.
- Consider skipping feature engineering entirely for a baseline — it establishes
  whether raw features already carry enough signal.

**Class imbalance** (when is_imbalanced=true):
- Tree ensembles: use class_weight="balanced" or scale_pos_weight=n_neg/n_pos.
- Threshold: the default 0.5 decision threshold may not be optimal — the model's
  probability output is what matters, threshold is tuned at deployment time.
- Metric: ROC-AUC is robust to imbalance; avg_precision is better when positives
  are very rare.

**Algorithm selection**:
- XGBoost / GradientBoosting: strong default for tabular data, handles missing
  values, built-in imbalance handling. Start here unless data is very small.
- RandomForest: more robust to outliers, good for noisy data, parallelizes well.
- LogisticRegression: fast linear baseline. Useful to establish whether the
  problem is linearly separable before adding complexity.
- Prefer simpler models when n_samples < 5,000 to avoid overfitting.

**Resampling** (resample_dataset tool) — data-level imbalance handling:
- Use when class_weight/scale_pos_weight alone isn't improving recall adequately,
  or when you want to test whether data-level vs algorithm-level imbalance handling
  works better for this dataset.
- ONLY resample the TRAIN split — never test. Resampling test data gives misleading metrics.
- "oversample": fast, no new information, good first try.
- "smote": synthetic samples via interpolation — more diverse than random oversampling,
  better for high-dimensional or sparse minority classes.
- "undersample": loses majority data — only useful when majority class is very large
  and training speed is a concern.

**Feature selection** (select_features tool) — prune after feature engineering:
- Use after engineer_features when the feature count is large (20+) and you suspect
  many features are redundant or noisy (e.g., rolling stats at many window sizes).
- "mutual_info": ranks by non-linear dependence with target — best general choice.
- "variance_threshold": drops near-constant features — cheap first pass.
- "correlation_filter": drops redundant features that are highly correlated with
  each other — useful when many rolling windows capture the same trend.
- Can be applied before or after splitting. Apply the same selection to both train
  and test to ensure the model sees the same features at evaluation time.

**Prediction output** (get_predictions tool) — enables two advanced patterns:
1. Error analysis: train a model → get_predictions(model, test_file, target) →
   explore_dataset(predictions_file, {{"class_distributions": ["feature_x"],
   "target_column": "correct"}}) to see which examples the model gets wrong.
   Use this to inform feature engineering for the next iteration.
2. Stacking: train base_model → get_predictions(base_model, train_file, target) →
   train a meta_model on the predictions CSV (use "predicted_prob" as a feature
   alongside original features) → evaluate meta_model on test.
   get_predictions returns a CSV with columns: all originals + predicted_prob,
   predicted_class, correct.

**Pipeline structure** — you are not required to follow a fixed sequence.
Design what makes sense for this specific experiment.

## Available tools

{tools_section}

## Monty sandbox restrictions

{monty_rules}

## Critical patterns for using tool results

split_dataset returns a File — call it twice:
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file  = split_dataset(data, target_column, 0.2, time_column, "test")

engineer_features returns a File — chain calls freely:
    eng = engineer_features(train_file, {{"rolling_columns": [...], "windows": [...]}})
    eng2 = engineer_features(eng, {{"normalize": true, "target_column": target_column}})

train_model returns a File — pass directly to evaluate_model:
    model_file = train_model(train_file, target_column, algorithm, hyperparams)
    eval_result = evaluate_model(model_file, test_file, target_column)

evaluate_model returns a dict — subscript reads are allowed:
    roc = eval_result["metrics"]["roc_auc"]

Do NOT use augmented assignment (+=), subscript assignment (d["k"]=v), or try/except.
Build dicts as literals only. The last expression (no assignment) is the return value.

## When fixing a previous error

Read the error and the failing code carefully before writing a fix. Identify the root
cause — do not just change variable names or add no-ops. Trace what each tool returns,
what each subsequent call expects, and where the mismatch is. Then fix the underlying
logic, not just the surface symptom.

## Pipeline design — you decide the structure

You are NOT required to follow a fixed sequence. Design the pipeline that makes most
sense for the experiment. Examples of valid approaches:

Baseline (no feature engineering):
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file = split_dataset(data, target_column, 0.2, time_column, "test")
    model_file = train_model(train_file, target_column, algorithm, hyperparams)
    eval_result = evaluate_model(model_file, test_file, target_column)
    {{"experiment_name": experiment_name, "algorithm": algorithm, "metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"], "threshold_analysis": eval_result["threshold_analysis"], "n_samples": eval_result["n_samples"]}}

Two-stage feature engineering (rolling then normalize separately):
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file = split_dataset(data, target_column, 0.2, time_column, "test")
    rolled_train = engineer_features(train_file, {{"rolling_columns": ["vibration"], "windows": [6, 24]}})
    rolled_test  = engineer_features(test_file,  {{"rolling_columns": ["vibration"], "windows": [6, 24]}})
    eng_train = engineer_features(rolled_train, {{"normalize": true, "target_column": target_column}})
    eng_test  = engineer_features(rolled_test,  {{"normalize": true, "target_column": target_column}})
    model_file = train_model(eng_train, target_column, algorithm, hyperparams)
    eval_result = evaluate_model(model_file, eng_test, target_column)
    {{"experiment_name": experiment_name, "algorithm": algorithm, "metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"], "threshold_analysis": eval_result["threshold_analysis"], "n_samples": eval_result["n_samples"]}}

Compare two class weightings and return the better model:
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file = split_dataset(data, target_column, 0.2, time_column, "test")
    model_a = train_model(train_file, target_column, "xgboost", {{"n_estimators": 100, "scale_pos_weight": 10}})
    model_b = train_model(train_file, target_column, "xgboost", {{"n_estimators": 100, "scale_pos_weight": 33}})
    eval_a = evaluate_model(model_a, test_file, target_column)
    eval_b = evaluate_model(model_b, test_file, target_column)
    best_eval = eval_a if eval_a["metrics"]["roc_auc"] > eval_b["metrics"]["roc_auc"] else eval_b
    {{"experiment_name": experiment_name, "algorithm": "xgboost", "metrics": best_eval["metrics"], "confusion_matrix": best_eval["confusion_matrix"], "threshold_analysis": best_eval["threshold_analysis"], "n_samples": best_eval["n_samples"]}}

SMOTE oversampling before training:
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file  = split_dataset(data, target_column, 0.2, time_column, "test")
    eng_train  = engineer_features(train_file, {{"rolling_columns": ["vibration_mms"], "windows": [6, 12]}})
    eng_test   = engineer_features(test_file,  {{"rolling_columns": ["vibration_mms"], "windows": [6, 12]}})
    resampled_train = resample_dataset(eng_train, target_column, {{"strategy": "smote", "target_ratio": 0.2}})
    model_file = train_model(resampled_train, target_column, algorithm, hyperparams)
    eval_result = evaluate_model(model_file, eng_test, target_column)
    {{"experiment_name": experiment_name, "algorithm": algorithm, "metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"], "threshold_analysis": eval_result["threshold_analysis"], "n_samples": eval_result["n_samples"]}}

Feature engineering followed by feature selection:
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file  = split_dataset(data, target_column, 0.2, time_column, "test")
    eng_train  = engineer_features(train_file, {{"rolling_columns": ["vibration_mms", "temperature_c"], "windows": [6, 12, 24]}})
    eng_test   = engineer_features(test_file,  {{"rolling_columns": ["vibration_mms", "temperature_c"], "windows": [6, 12, 24]}})
    sel_train  = select_features(eng_train, target_column, {{"method": "mutual_info", "k": 15}})
    sel_test   = select_features(eng_test,  target_column, {{"method": "mutual_info", "k": 15}})
    model_file = train_model(sel_train, target_column, algorithm, hyperparams)
    eval_result = evaluate_model(model_file, sel_test, target_column)
    {{"experiment_name": experiment_name, "algorithm": algorithm, "metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"], "threshold_analysis": eval_result["threshold_analysis"], "n_samples": eval_result["n_samples"]}}

Error analysis — explore what the model gets wrong, then return that as insight:
    train_file = split_dataset(data, target_column, 0.2, time_column, "train")
    test_file  = split_dataset(data, target_column, 0.2, time_column, "test")
    model_file = train_model(train_file, target_column, algorithm, hyperparams)
    pred_file  = get_predictions(model_file, test_file, target_column)
    error_analysis = explore_dataset(pred_file, {{"target_column": "correct", "class_distributions": ["vibration_mms", "temperature_c"]}})
    eval_result = evaluate_model(model_file, test_file, target_column)
    {{"experiment_name": experiment_name, "algorithm": algorithm, "metrics": eval_result["metrics"], "confusion_matrix": eval_result["confusion_matrix"], "threshold_analysis": eval_result["threshold_analysis"], "n_samples": eval_result["n_samples"], "error_analysis": error_analysis}}

The last expression MUST be a dict with at minimum these keys:
    experiment_name, algorithm, metrics, confusion_matrix, threshold_analysis, n_samples
Additional keys (e.g. error_analysis) are allowed and will appear in the report.

## Response format

Respond in exactly this format:

## Reasoning
[Your thinking: what pipeline makes sense for this experiment and why. Consider whether
feature engineering helps, whether class imbalance needs special treatment, whether
chaining multiple steps adds value, etc.]

## Code
```python
[your orchestration code]
```
"""


def _build_analysis_system_prompt(max_iterations: int, current_iteration: int) -> str:
    remaining = max_iterations - current_iteration - 1
    return f"""\
You are an expert ML engineer analyzing experiment results to guide the next iteration
of model development.

You must respond with valid JSON only — no markdown, no explanation outside the JSON.

Response format:
{{
  "should_continue": true | false,
  "reasoning": "What you observed, what it tells you, and what to try next",
  "exploration_requests": [
    {{
      "question": "The specific hypothesis you are testing, e.g. 'Do failure cases show meaningfully higher vibration than healthy cases?'",
      "analysis_type": "class_distributions",
      "target_column": "failure_24h",
      "class_distributions": ["vibration_mms", "temperature_c"]
    }}
  ],
  "next_experiments": [
    {{
      "name": "descriptive experiment name",
      "algorithm": "xgboost" | "random_forest" | "gradient_boosting" | "logistic_regression",
      "hyperparams": {{ ... algorithm-specific hyperparams ... }},
      "feature_config": {{
        "group_column": "...",
        "time_column": "...",
        "rolling_columns": [...],
        "windows": [...],
        "lag_columns": [...],
        "lags": [...],
        "normalize": true | false,
        "drop_columns": [...],
        "fillna_method": "forward"
      }},
      "rationale": "Why this experiment is worth trying"
    }}
  ]
}}

exploration_requests rules:
- Max 2 requests per iteration.
- Each request targets EXACTLY ONE analysis_type. Do not mix multiple types in one request.
- Supported analysis_type values and their required config fields:
    "class_distributions" → requires: target_column, class_distributions (list of columns)
    "correlation_matrix"  → requires: correlation_matrix: true
    "temporal_trend"      → requires: temporal_trend: {{time_column, target_column, freq}}
    "group_stats"         → requires: group_stats: {{group_column, target_column}}
    "outlier_summary"     → requires: outlier_summary (list of columns)
    "feature_target_corr_by_group" → requires: feature_target_corr_by_group: {{group_column, target_column, feature_columns}}
- The "question" field is required. It must be a specific testable hypothesis, not a
  general request. Bad: "explore the data". Good: "Is vibration_mms higher for failures?"
- Set exploration_requests to [] if the current results already tell you enough to
  design the next experiments. Only explore when you have a concrete unanswered question.

When deciding next experiments, reason about WHAT WAS TRIED vs what hasn't been explored.
Each result includes used_feature_engineering, used_rolling_features, used_lag_features.
Think systematically: if no feature engineering was tried yet, does the data profile
suggest it would help (weak raw correlations, temporal/sequential structure)?
If feature engineering helped, can it be improved? Avoid experiments identical to ones tried.

Iteration context: this is iteration {current_iteration + 1} of {max_iterations} requested.
Remaining iterations allowed: {remaining}.
Set should_continue=false only if:
- Best ROC-AUC >= 0.97, OR
- No remaining iterations (remaining == 0), OR
- Results have genuinely plateaued (< 0.005 ROC-AUC improvement over last iteration
  AND you have already tried the most promising directions)
Otherwise keep exploring — the user asked for {max_iterations} iterations for a reason.
"""


def _build_initial_design_system_prompt() -> str:
    return """\
You are an expert ML engineer. Given a dataset profile and a problem description,
design the first batch of experiments to run.

You must respond with valid JSON only — no markdown, no explanation outside the JSON.

Response format:
{
  "problem_type": "binary_classification",
  "primary_metric": "roc_auc" | "f1" | "recall",
  "reasoning": "Brief description of your strategy",
  "experiments": [
    {
      "name": "descriptive experiment name",
      "algorithm": "xgboost" | "random_forest" | "gradient_boosting" | "logistic_regression",
      "hyperparams": { ... algorithm-specific hyperparams ... },
      "feature_config": {
        "group_column": "",
        "time_column": "",
        "rolling_columns": [],
        "windows": [],
        "lag_columns": [],
        "lags": [],
        "normalize": false,
        "drop_columns": [],
        "fillna_method": "forward"
      },
      "rationale": "Why this experiment makes sense given the data profile"
    }
  ]
}

Design 2-3 experiments for the first batch. Good first batches typically include:
- A fast baseline to establish a floor (e.g. logistic_regression with default settings)
- Your best initial hypothesis given the data profile
- Optionally one variant that tests a specific idea suggested by the profile

Use the dataset profile to guide your choices:
- feature_target_corr: weak raw correlations suggest feature engineering may help
- categorical_columns: note these are excluded from the model automatically
- is_imbalanced: handle with class_weight or scale_pos_weight
- Shape and column types should inform algorithm complexity (simpler models for small datasets)
- A time column suggests sequential structure; lag/rolling features may capture temporal patterns

The feature_config in each experiment describes what engineer_features should apply.
Leave all fields empty/false if no feature engineering is needed for that experiment.
The orchestration code generator will decide the exact pipeline — your job here is
to specify what the experiment is trying to learn, not to prescribe every implementation detail.
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def _call_llm(system: str, messages: list[dict], model: str = "gpt-4o") -> str:
    """Call OpenAI and return the response text."""
    client = _openai_client()
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "system", "content": system}, *messages],
        temperature=0.2,
    )
    return response.choices[0].message.content


def _extract_code(text: str) -> str:
    """Pull Python code out of a markdown code block."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    return text.strip()


def _extract_reasoning(text: str) -> str:
    """Extract the ## Reasoning section from LLM response."""
    if "## Reasoning" in text:
        start = text.index("## Reasoning") + len("## Reasoning")
        if "## Code" in text:
            end = text.index("## Code")
            return text[start:end].strip()
        return text[start:].strip()
    return ""


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _recommend_threshold(threshold_analysis: list, min_precision: float = 0.70) -> dict | None:
    """Find the threshold that maximises recall subject to precision >= min_precision."""
    candidates = [t for t in threshold_analysis if t["precision"] >= min_precision]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t["recall"])


def _print_experiment_table(results: list["ExperimentResult"], best_name: str) -> None:
    """Print a ranked comparison table of all experiments."""
    sorted_results = sorted(results, key=lambda r: r.metrics.get("roc_auc", 0), reverse=True)
    print("\n" + "─" * 78)
    print(f"  {'Rank':<5} {'Experiment':<32} {'ROC-AUC':<9} {'F1':<7} {'Recall':<8} {'Note'}")
    print("─" * 78)
    for rank, r in enumerate(sorted_results, 1):
        note = "◀ winner" if r.name == best_name else ""
        roc = r.metrics.get("roc_auc", 0)
        f1 = r.metrics.get("f1", 0)
        recall = r.metrics.get("recall", 0)
        print(f"  {rank:<5} {r.name:<32} {roc:<9.4f} {f1:<7.4f} {recall:<8.4f} {note}")
    print("─" * 78)


def _print_threshold_recommendation(threshold_analysis: list, default_metrics: dict) -> None:
    """Print the operational threshold recommendation."""
    rec = _recommend_threshold(threshold_analysis)
    if not rec:
        return
    default_recall = default_metrics.get("recall", 0)
    default_precision = default_metrics.get("precision", 0)
    missed_pct = round((1 - rec["recall"]) * 100, 1)
    false_alarm_pct = round((1 - rec["precision"]) * 100, 1)

    print(f"\n  Recommended decision threshold: {rec['threshold']}")
    print(f"  ├─ Precision : {rec['precision']:.0%}   ({false_alarm_pct}% of alerts are false alarms)")
    print(f"  ├─ Recall    : {rec['recall']:.0%}   (catches {rec['recall']*100:.0f}% of actual failures)")
    print(f"  └─ F1        : {rec['f1']:.4f}")
    print(f"  Default threshold (0.5): Precision={default_precision:.0%}, Recall={default_recall:.0%}")
    if rec["recall"] > default_recall:
        extra = round((rec["recall"] - default_recall) * 100, 1)
        print(f"  → Lowering threshold catches {extra}% more failures at cost of more alerts")


# ---------------------------------------------------------------------------
# Orchestration code generation (durable Flyte task with Flyte report)
# ---------------------------------------------------------------------------

@agent_env.task
async def plan_experiment(
    experiment_json: str,
    profile_json: str,
    target_column: str,
    time_column: str,
    previous_error: str = "",
    previous_code: str = "",
    llm_model: str = "gpt-4o",
) -> str:
    """LLM plans a single experiment: reasons about the pipeline and generates Monty code.

    Runs as a durable Flyte task so each experiment's planning step is traceable.
    Returns a JSON string: {"code": "...", "reasoning": "..."}.

    Args:
        experiment_json: JSON string of the experiment spec (name, algorithm, hyperparams, ...).
        profile_json: JSON string of the dataset profile from profile_dataset.
        target_column: Name of the target column.
        time_column: Time column for temporal splitting, or empty string.
        previous_error: Error message from the previous attempt (empty on first try).
        previous_code: Code that failed on the previous attempt (empty on first try).
        llm_model: OpenAI model identifier.

    Returns:
        str — JSON string with keys "code" and "reasoning".
    """
    experiment = json.loads(experiment_json)
    profile = json.loads(profile_json)
    exp_name = experiment.get("name", "experiment")

    # Strip rationale — it was written by the design LLM to explain *why* this
    # experiment was chosen. Passing it here causes plan_experiment to parrot it
    # back as "reasoning" instead of independently thinking about *how* to build
    # the best pipeline. Keep only the technical spec.
    pipeline_spec = {
        k: v for k, v in experiment.items()
        if k not in ("rationale",)
    }

    system = _build_orchestration_system_prompt(profile)

    user_content = textwrap.dedent(f"""
        Design and implement the best pipeline for this experiment:

        Name: {exp_name}
        Algorithm: {pipeline_spec.get("algorithm")}
        Hyperparams: {json.dumps(pipeline_spec.get("hyperparams", {}), indent=2)}
        Feature config hint: {json.dumps(pipeline_spec.get("feature_config", {}), indent=2)}

        Available sandbox inputs:
        - data: File  — the full dataset CSV
        - target_column: str = "{target_column}"
        - time_column: str = "{time_column}"  (empty string means no time ordering)
        - experiment_name: str = "{exp_name}"

        The feature config hint is a suggestion from the experiment designer — you can
        follow it, improve on it, or override it if the dataset context and your ML
        judgment suggest a better approach. In your ## Reasoning, explain your actual
        pipeline decisions: what you chose to do (or not do) and why, based on the
        dataset profile above. Do not restate the experiment name or why it was chosen.
    """).strip()

    messages = [{"role": "user", "content": user_content}]
    if previous_code and previous_error:
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"```python\n{previous_code}\n```"},
            {"role": "user", "content": f"That code failed with this error:\n\n{previous_error}\n\nPlease fix it."},
        ]

    response = await _call_llm(system, messages, llm_model)
    reasoning = _extract_reasoning(response)
    code = _extract_code(response)
    return json.dumps({"code": code, "reasoning": reasoning})


@flyte.trace
async def design_experiments(
    problem_description: str,
    profile_json: str,
    llm_model: str = "gpt-4o",
) -> str:
    """LLM designs the initial batch of experiments given problem + dataset profile.

    Traced so the prompt/response is visible in the Flyte UI and results are
    cached for deterministic replay on crash/retry.
    Returns raw LLM response (JSON string matching InitialDesign schema).
    """
    design_prompt = textwrap.dedent(f"""
        Problem description: {problem_description}

        Dataset profile:
        {profile_json}

        Design the first batch of experiments.
    """).strip()
    return await _call_llm(
        _build_initial_design_system_prompt(),
        [{"role": "user", "content": design_prompt}],
        llm_model,
    )


@flyte.trace
async def analyze_iteration(
    analysis_prompt: str,
    max_iterations: int,
    current_iteration: int,
    llm_model: str = "gpt-4o",
) -> str:
    """LLM analyzes experiment results and decides whether/how to continue.

    Traced so the prompt/response is visible in the Flyte UI and results are
    cached for deterministic replay on crash/retry.
    Returns raw LLM response (JSON string matching IterationDecision schema).
    """
    return await _call_llm(
        _build_analysis_system_prompt(max_iterations, current_iteration),
        [{"role": "user", "content": analysis_prompt}],
        llm_model,
    )


@flyte.trace
async def plan_followup(
    analysis_prompt: str,
    analysis_response: str,
    followup_prompt: str,
    max_iterations: int,
    current_iteration: int,
    llm_model: str = "gpt-4o",
) -> str:
    """LLM designs next experiments after targeted data explorations.

    Traced so the prompt/response is visible in the Flyte UI and results are
    cached for deterministic replay on crash/retry.
    Returns raw LLM response (JSON string with {"next_experiments": [...]}).
    """
    return await _call_llm(
        _build_analysis_system_prompt(max_iterations, current_iteration),
        [
            {"role": "user", "content": analysis_prompt},
            {"role": "assistant", "content": analysis_response},
            {"role": "user", "content": followup_prompt},
        ],
        llm_model,
    )


def _corrupt_experiment_for_demo(exp_dict: dict) -> dict:
    """Introduce a deliberate error into the first experiment for demo purposes.

    Corrupts the algorithm name so the LLM must recover from a known-bad value.
    The retry loop will catch this, regenerate with the error message, and fix it.
    """
    corrupted = dict(exp_dict)
    corrupted["algorithm"] = corrupted["algorithm"] + "_INVALID"
    return corrupted


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    name: str
    algorithm: str
    metrics: dict
    confusion_matrix: dict
    threshold_analysis: list
    n_samples: int
    code: str
    attempts: int
    reasoning: str = ""
    error: str = ""


@dataclass
class AgentResult:
    model_card: str
    best_experiment: str
    best_metrics: dict
    all_results: list[ExperimentResult]
    iterations: int
    total_experiments: int


async def _run_experiment(
    exp: "ExperimentConfig",
    exp_dict: dict,
    inject_failure: bool,
    data: File,
    target_column: str,
    time_column: str,
    profile: dict,
    llm_model: str,
    max_retries: int,
) -> "ExperimentResult | None":
    """Run a single experiment with retries. Returns None on total failure."""
    exp_name = exp.name
    profile_json = json.dumps(profile)
    print(f"\n   ┌─ {exp_name}  [{exp.algorithm}]")
    if exp.rationale:
        for line in textwrap.wrap(exp.rationale, width=58):
            print(f"   │  {line}")
    if inject_failure:
        print(f"   │  [injecting failure for demo: algorithm='{exp_dict['algorithm']}']")

    code = ""
    error = ""
    result = None
    attempt = 0

    reasoning = ""
    for attempt in range(max_retries):
        try:
            with flyte.group(exp_name):
                plan_json = await plan_experiment.aio(
                    experiment_json=json.dumps(exp_dict),
                    profile_json=profile_json,
                    target_column=target_column,
                    time_column=time_column,
                    previous_error=error,
                    previous_code=code,
                    llm_model=llm_model,
                )
                plan = json.loads(plan_json)
                code = plan["code"]
                reasoning = plan.get("reasoning", "")
                result = await flyte.sandbox.orchestrate_local(
                    code,
                    inputs={"data": data, "target_column": target_column,
                            "time_column": time_column, "experiment_name": exp_name},
                    tasks=TOOLS,
                )
            error = ""
            break
        except Exception as exc:
            error = str(exc)
            short_error = error[:100] + "..." if len(error) > 100 else error
            print(f"   │  attempt {attempt + 1} failed: {short_error}")
            print(f"   │  → asking LLM to fix and retry...")
            if inject_failure and attempt == 0:
                exp_dict = exp.model_dump()

    if result and not error:
        exp_result = ExperimentResult(
            name=exp_name,
            algorithm=exp.algorithm,
            metrics=result.get("metrics", {}),
            confusion_matrix=result.get("confusion_matrix", {}),
            threshold_analysis=result.get("threshold_analysis", []),
            n_samples=result.get("n_samples", 0),
            code=code,
            reasoning=reasoning,
            attempts=attempt + 1,
        )
        m = exp_result.metrics
        attempts_note = f" (recovered after {attempt + 1} attempts)" if attempt > 0 else ""
        print(f"   └─ ROC-AUC={m.get('roc_auc')}, F1={m.get('f1')}, Recall={m.get('recall')}{attempts_note}")
        return exp_result

    print(f"   └─ FAILED after {max_retries} attempts — skipping.")
    return None


async def run_agent(
    data: File,
    problem_description: str,
    target_column: str,
    time_column: str = "",
    max_iterations: int = 3,
    max_retries_per_experiment: int = 3,
    llm_model: str = "gpt-4o",
    inject_failure: bool = False,
) -> AgentResult:
    """Run the MLE agent end-to-end.

    Args:
        data: CSV file containing the dataset.
        problem_description: Natural language description of the ML problem.
        target_column: Name of the target column to predict.
        time_column: Optional column to use for time-based train/test split.
        max_iterations: Maximum number of experiment iterations to run.
        max_retries_per_experiment: Max times to retry a failed sandbox execution.
        llm_model: OpenAI model to use (default: gpt-4o).
        inject_failure: If True, corrupts the first experiment to demonstrate self-healing.
    """
    print(f"\n{'='*60}")
    print(f"MLE Agent starting")
    print(f"Problem: {problem_description}")
    print(f"Target: {target_column}")
    if inject_failure:
        print(f"[demo mode: failure injection enabled]")
    print(f"{'='*60}\n")

    # --- Phase 1: Profile the dataset (trusted tool, LLM never sees raw data) ---
    print(">> Phase 1: Profiling dataset...")
    with flyte.group("profile"):
        profile = await profile_dataset(data, target_column)
    print(f"   Shape: {profile['shape']}, Classes: {profile['target_distribution']}")
    print(f"   Imbalanced: {profile['is_imbalanced']}, Columns: {len(profile['columns'])}")
    corr = profile.get("feature_target_corr", {})
    top_corr = list(corr.items())[:5]
    print(f"   Top correlations: {', '.join(f'{k}={v:+.3f}' for k,v in top_corr)}")

    # Stream report: dataset summary
    await flyte.report.log.aio(
        f"<h1>MLE Agent Run</h1>"
        f"<p><b>Problem:</b> {problem_description}</p>"
        f"<p><b>Dataset:</b> {profile['shape'][0]:,} rows × {profile['shape'][1]} cols &nbsp;|&nbsp; "
        f"Class balance: {profile['class_balance']} &nbsp;|&nbsp; Imbalanced: {profile['is_imbalanced']}</p>"
        f"<p><b>Top feature-target correlations (raw):</b> "
        + ", ".join(f"{k}: {v:+.3f}" for k, v in top_corr) +
        f"</p><hr>",
        do_flush=True,
    )

    # --- Phase 2: LLM designs initial experiments ---
    print("\n>> Phase 2: Designing initial experiments...")
    design_response = await design_experiments(
        problem_description=problem_description,
        profile_json=json.dumps(profile),
        llm_model=llm_model,
    )
    design = InitialDesign.model_validate(_parse_json(design_response))
    print(f"   Primary metric: {design.primary_metric}")
    print(f"   Strategy: {design.reasoning}")
    print(f"   Experiments planned: {len(design.experiments)}")

    all_results: list[ExperimentResult] = []
    iteration_log: list[dict] = []  # tracks per-iteration decisions + explorations for summary
    current_experiments: list[ExperimentConfig] = design.experiments
    first_experiment = True

    # --- Phase 3: Iterative experiment loop ---
    for iteration in range(max_iterations):
        experiments = current_experiments

        if not experiments:
            print(f"\n>> No experiments to run in iteration {iteration + 1}. Stopping.")
            break

        print(f"\n>> Phase 3.{iteration + 1}: Running {len(experiments)} experiment(s) in parallel...")

        # Assign names and prepare dicts before launching in parallel
        exp_batch = []
        for i, exp in enumerate(experiments):
            if not exp.name:
                exp.name = f"experiment_{len(all_results) + i + 1}"
            exp_dict = exp.model_dump()
            inject_this = inject_failure and first_experiment and i == 0
            if inject_this:
                exp_dict = _corrupt_experiment_for_demo(exp_dict)
            first_experiment = False
            exp_batch.append((exp, exp_dict, inject_this))

        batch_results = await asyncio.gather(*[
            _run_experiment(
                exp=exp,
                exp_dict=exp_dict,
                inject_failure=inject_this,
                data=data,
                target_column=target_column,
                time_column=time_column,
                profile=profile,
                llm_model=llm_model,
                max_retries=max_retries_per_experiment,
            )
            for exp, exp_dict, inject_this in exp_batch
        ])

        for exp_result in batch_results:
            if exp_result is not None:
                all_results.append(exp_result)
                # Stream report: each experiment as it completes
                m = exp_result.metrics
                html = (
                    f"<h3>Iteration {iteration + 1} — {exp_result.name}</h3>"
                    f"<p><b>Algorithm:</b> {exp_result.algorithm} &nbsp;|&nbsp; "
                    f"<b>ROC-AUC:</b> {m.get('roc_auc')} &nbsp;|&nbsp; "
                    f"<b>F1:</b> {m.get('f1')} &nbsp;|&nbsp; "
                    f"<b>Recall:</b> {m.get('recall')} &nbsp;|&nbsp; "
                    f"<b>Attempts:</b> {exp_result.attempts}</p>"
                )
                if exp_result.reasoning:
                    html += f"<details><summary>Reasoning</summary><pre>{exp_result.reasoning}</pre></details>"
                html += f"<details><summary>Generated Code</summary><pre>{exp_result.code}</pre></details>"
                await flyte.report.log.aio(html, do_flush=True)

        # --- Phase 4: Analyze results, decide whether to iterate ---
        if all_results and iteration < max_iterations - 1:
            print(f"\n>> Phase 4.{iteration + 1}: Analyzing results, deciding next steps...")
            results_summary = [
                {
                    "experiment_name": r.name,
                    "algorithm": r.algorithm,
                    "metrics": r.metrics,
                    "confusion_matrix": r.confusion_matrix,
                    "used_feature_engineering": "engineer_features" in r.code,
                    "used_rolling_features": "rolling_columns" in r.code,
                    "used_lag_features": "lag_columns" in r.code,
                }
                for r in all_results
            ]
            analysis_prompt = textwrap.dedent(f"""
                Problem: {problem_description}
                Dataset profile: shape={profile['shape']}, imbalanced={profile['is_imbalanced']}
                Feature-target correlations (raw): {json.dumps(profile.get('feature_target_corr', {}), indent=2)}

                Experiment results so far (iteration {iteration + 1}):
                {json.dumps(results_summary, indent=2)}

                Should we run more experiments? If yes, request any data explorations
                you need, then specify what experiments to run next.
            """).strip()

            analysis_response = await analyze_iteration(
                analysis_prompt=analysis_prompt,
                max_iterations=max_iterations,
                current_iteration=iteration,
                llm_model=llm_model,
            )
            decision = IterationDecision.model_validate(_parse_json(analysis_response))
            verdict = "continuing" if decision.should_continue else "stopping"
            print(f"   Decision: {verdict}")
            print(f"   Reasoning: {decision.reasoning}")

            # Stream report: analysis decision
            await flyte.report.log.aio(
                f"<h3>Analysis — Iteration {iteration + 1}</h3>"
                f"<p><b>Decision:</b> {verdict}</p>"
                f"<p><b>Reasoning:</b> {decision.reasoning}</p>",
                do_flush=True,
            )

            # Track this iteration for the experiment journey summary
            iter_entry = {
                "iteration": iteration + 1,
                "experiments": [r.name for r in batch_results if r is not None],
                "best_roc_auc": max(
                    (r.metrics.get("roc_auc", 0) for r in all_results), default=0
                ),
                "reasoning": decision.reasoning,
                "explorations": [],
            }

            # --- Targeted exploration before next iteration ---
            if decision.should_continue and decision.exploration_requests:
                print(f"   Running {len(decision.exploration_requests)} exploration request(s)...")
                exploration_questions = []
                exploration_results = []

                for i, req in enumerate(decision.exploration_requests):
                    question = req.get("question", f"Exploration {i + 1}")
                    # Strip agent-level metadata — tool only needs the analysis config
                    tool_config = {k: v for k, v in req.items() if k not in ("question", "analysis_type")}

                    print(f"   Q: {question}")
                    with flyte.group(f"explore_{iteration + 1}_{i + 1}"):
                        result = await explore_dataset(data, tool_config)
                    exploration_questions.append(question)
                    exploration_results.append(result)
                    iter_entry["explorations"].append({"question": question})

                    await flyte.report.log.aio(
                        f"<h4>Exploration {i + 1}</h4>"
                        f"<p><b>Question:</b> {question}</p>"
                        f"<details><summary>Results</summary><pre>{json.dumps(result, indent=2)}</pre></details>",
                        do_flush=True,
                    )

                # Build follow-up that explicitly connects each question to its answer
                qa_pairs = "\n\n".join(
                    f'Question {i + 1}: "{q}"\nResult:\n{json.dumps(r, indent=2)}'
                    for i, (q, r) in enumerate(zip(exploration_questions, exploration_results))
                )
                followup_prompt = textwrap.dedent(f"""
                    You requested {len(exploration_results)} targeted exploration(s).
                    Here is what you asked and what you learned:

                    {qa_pairs}

                    Given what you learned and your earlier reasoning:
                    "{decision.reasoning}"

                    Now specify the next experiments. For each experiment, briefly state
                    which exploration insight informed your choice.
                    Respond with valid JSON: {{"next_experiments": [...same schema as before...]}}
                """).strip()
                followup_response = await plan_followup(
                    analysis_prompt=analysis_prompt,
                    analysis_response=analysis_response,
                    followup_prompt=followup_prompt,
                    max_iterations=max_iterations,
                    current_iteration=iteration,
                    llm_model=llm_model,
                )
                followup = _parse_json(followup_response)
                current_experiments = IterationDecision.model_validate({
                    "should_continue": True,
                    "reasoning": decision.reasoning,
                    "next_experiments": followup.get("next_experiments", []),
                }).next_experiments
                print(f"   Post-exploration: {len(current_experiments)} experiment(s) planned")
            else:
                current_experiments = decision.next_experiments

            iteration_log.append(iter_entry)

            if not decision.should_continue:
                break

    # --- Phase 5: Rank all results and generate model card ---
    print(f"\n>> Phase 5: Ranking {len(all_results)} experiment(s) and generating model card...")

    if not all_results:
        return AgentResult(
            model_card="No experiments completed successfully.",
            best_experiment="",
            best_metrics={},
            all_results=[],
            iterations=iteration + 1,
            total_experiments=0,
        )

    ranking_input = [
        {
            "experiment_name": r.name,
            "metrics": r.metrics,
            "confusion_matrix": r.confusion_matrix,
        }
        for r in all_results
    ]
    with flyte.group("rank"):
        ranking = await rank_experiments(json.dumps(ranking_input))
    best_name = ranking["best_experiment"]
    best_result = next(r for r in all_results if r.name == best_name)

    _print_experiment_table(all_results, best_name)
    _print_threshold_recommendation(best_result.threshold_analysis, best_result.metrics)

    # Stream report: final rankings table
    rows = "".join(
        f"<tr><td>{row['rank']}</td>"
        f"<td>{'<b>' if row['experiment_name'] == best_name else ''}"
        f"{row['experiment_name']}"
        f"{'</b>' if row['experiment_name'] == best_name else ''}</td>"
        f"<td>{row['roc_auc']}</td><td>{row['f1']}</td>"
        f"<td>{row['recall']}</td><td>{row['precision']}</td></tr>"
        for row in ranking.get("ranking", [])
    )
    await flyte.report.log.aio(
        f"<hr><h2>Final Rankings</h2>"
        f"<table border='1' cellpadding='6' cellspacing='0'>"
        f"<tr><th>Rank</th><th>Experiment</th><th>ROC-AUC</th><th>F1</th><th>Recall</th><th>Precision</th></tr>"
        f"{rows}</table>"
        f"<p>{ranking.get('summary', '')}</p>",
        do_flush=True,
    )

    # Stream report: experiment journey summary
    journey_rows = ""
    for entry in iteration_log:
        exps = ", ".join(entry["experiments"]) if entry["experiments"] else "—"
        explorations = "; ".join(e["question"] for e in entry["explorations"]) if entry["explorations"] else "—"
        short_reasoning = (entry["reasoning"][:120] + "…") if len(entry["reasoning"]) > 120 else entry["reasoning"]
        journey_rows += (
            f"<tr>"
            f"<td style='text-align:center'>{entry['iteration']}</td>"
            f"<td>{exps}</td>"
            f"<td style='text-align:center'>{entry['best_roc_auc']:.4f}</td>"
            f"<td>{short_reasoning}</td>"
            f"<td>{explorations}</td>"
            f"</tr>"
        )
    await flyte.report.log.aio(
        f"<hr><h2>Experiment Journey</h2>"
        f"<table border='1' cellpadding='6' cellspacing='0' style='width:100%;border-collapse:collapse'>"
        f"<tr><th>Iter</th><th>Experiments</th><th>Best ROC-AUC</th><th>Key insight</th><th>Explorations</th></tr>"
        f"{journey_rows}"
        f"</table>",
        do_flush=True,
    )

    model_card = await _generate_model_card(
        problem_description=problem_description,
        profile=profile,
        all_results=all_results,
        best_result=best_result,
        ranking=ranking,
        iteration_log=iteration_log,
        llm_model=llm_model,
    )

    print(f"\n{'='*60}")
    print(f"DONE — Best model: {best_name}")
    print(f"       ROC-AUC={best_result.metrics.get('roc_auc')}, F1={best_result.metrics.get('f1')}")
    print(f"{'='*60}\n")

    return AgentResult(
        model_card=model_card,
        best_experiment=best_name,
        best_metrics=best_result.metrics,
        all_results=all_results,
        iterations=iteration + 1,
        total_experiments=len(all_results),
    )


async def _generate_model_card(
    problem_description: str,
    profile: dict,
    all_results: list[ExperimentResult],
    best_result: ExperimentResult,
    ranking: dict,
    iteration_log: list[dict],
    llm_model: str,
) -> str:
    """Generate a markdown model card summarizing the winning model."""
    system = textwrap.dedent("""
        You are an ML engineer writing a model card for a trained model.
        Write in markdown. Be concise but informative. Include:
        - Problem statement
        - Dataset summary
        - Experiment journey (brief per-iteration narrative: what was tried, what was learned, what changed)
        - Experiment summary (table of all experiments with metrics)
        - Winning model details (algorithm, key hyperparams, metrics, threshold analysis)
        - Recommendations for deployment (decision threshold, monitoring)
    """).strip()

    results_text = "\n".join(
        f"- {r.name} ({r.algorithm}): ROC-AUC={r.metrics.get('roc_auc')}, "
        f"F1={r.metrics.get('f1')}, Recall={r.metrics.get('recall')}"
        for r in all_results
    )

    journey_text = ""
    if iteration_log:
        journey_text = "\n\nIteration log:\n" + "\n".join(
            f"  Iteration {e['iteration']}: ran [{', '.join(e['experiments'])}], "
            f"best ROC-AUC so far={e['best_roc_auc']:.4f}. "
            f"Key insight: {e['reasoning'][:200]}. "
            + (f"Explorations: {'; '.join(x['question'] for x in e['explorations'])}" if e['explorations'] else "")
            for e in iteration_log
        )

    user_content = textwrap.dedent(f"""
        Problem: {problem_description}

        Dataset: {profile['shape'][0]} rows × {profile['shape'][1]} cols.
        Class balance: {profile['class_balance']}
        Imbalanced: {profile['is_imbalanced']}
        {journey_text}

        All experiments:
        {results_text}

        Best model: {best_result.name} ({best_result.algorithm})
        Metrics: {json.dumps(best_result.metrics, indent=2)}
        Confusion matrix: {json.dumps(best_result.confusion_matrix, indent=2)}
        Threshold analysis: {json.dumps(best_result.threshold_analysis, indent=2)}

        Ranking summary: {ranking['summary']}
    """).strip()

    response = await _call_llm(system, [{"role": "user", "content": user_content}], llm_model)
    return response


# ---------------------------------------------------------------------------
# Durable entrypoint (runs the agent as a Flyte task in the cloud)
# ---------------------------------------------------------------------------

@agent_env.task(retries=1, report=True)
async def mle_agent_task(
    data: File,
    problem_description: str,
    target_column: str,
    time_column: str = "",
    max_iterations: int = 3,
) -> str:
    """Durable Flyte task entrypoint for the MLE agent.

    This is the task you submit to Flyte for cloud execution.
    It runs the full agent loop and returns the model card as a string.
    """
    result = await run_agent(
        data=data,
        problem_description=problem_description,
        target_column=target_column,
        time_column=time_column,
        max_iterations=max_iterations,
    )
    return result.model_card
