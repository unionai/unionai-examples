"""Model evaluation tools.

Evaluate a trained model on held-out data, and rank multiple experiment results.
These tools run as durable Flyte tasks so evaluations are reproducible and traceable.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def evaluate_model(model: File, data: File, target_column: str) -> dict:
    """Evaluate a trained model on a dataset and return comprehensive metrics.

    Use this on the test split (data the model has never seen) to get
    an unbiased estimate of real-world performance.

    Args:
        model: Serialized model file produced by train_model.
        data: CSV file to evaluate on (e.g. the test split from split_dataset).
        target_column: Name of the target column in the data file.

    Returns a dict with keys:
        - algorithm: str — which algorithm was used
        - hyperparams: dict — hyperparams used during training
        - feature_columns: list[str] — features the model was trained on
        - n_samples: int — number of rows evaluated
        - metrics: dict with:
            - accuracy: float
            - f1: float (binary, positive class)
            - precision: float (binary, positive class)
            - recall: float (binary, positive class) — fraction of actual positives caught
            - roc_auc: float — area under ROC curve (1.0 = perfect, 0.5 = random)
            - avg_precision: float — area under precision-recall curve (better for imbalanced data)
        - confusion_matrix: {tn, fp, fn, tp} — true/false positive/negative counts
        - classification_report: str — sklearn text report with per-class metrics
        - threshold_analysis: list of {threshold, precision, recall, f1} at 10 thresholds
    """
    import joblib
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model_path = await model.download()
    data_path = await data.download()

    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_columns"]

    df = pd.read_csv(data_path)
    # Use only the features the model was trained on (in the same order)
    X = df[feature_cols].values
    y = df[target_column].values

    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(cm[0, 0]))

    # Threshold analysis: precision/recall tradeoff at multiple decision thresholds
    thresholds = [round(t, 1) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    threshold_analysis = []
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        threshold_analysis.append({
            "threshold": thresh,
            "precision": round(float(precision_score(y, y_pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred_t, zero_division=0)), 4),
            "f1": round(float(f1_score(y, y_pred_t, zero_division=0)), 4),
        })

    return {
        "algorithm": artifact.get("model").__class__.__name__,
        "hyperparams": {},  # stored in train result; retrieve from there
        "feature_columns": feature_cols,
        "n_samples": int(len(y)),
        "metrics": {
            "accuracy": round(float(accuracy_score(y, y_pred)), 4),
            "f1": round(float(f1_score(y, y_pred, average="binary", zero_division=0)), 4),
            "precision": round(float(precision_score(y, y_pred, average="binary", zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred, average="binary", zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
            "avg_precision": round(float(average_precision_score(y, y_prob)), 4),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "classification_report": classification_report(y, y_pred, zero_division=0),
        "threshold_analysis": threshold_analysis,
    }


@tool_env.task
async def rank_experiments(results_json: str) -> dict:
    """Rank a list of experiment results and identify the best model.

    Use this after collecting multiple evaluate_model results to pick the winner
    and understand the relative performance of different approaches.

    Primary ranking criterion: roc_auc (robust to class imbalance).
    Secondary criterion: f1 (balances precision and recall).

    Args:
        results_json: JSON string — list of dicts, each with at minimum:
                      {"experiment_name": str, "metrics": {roc_auc, f1, ...}}.

    Returns a dict with keys:
        - best_experiment: str — name of the best experiment
        - best_metrics: dict — metrics of the best experiment
        - ranking: list of {rank, experiment_name, roc_auc, f1, recall, precision}
                   sorted best-to-worst by roc_auc
        - summary: str — human-readable comparison paragraph
    """
    import json
    results = json.loads(results_json)
    if not results:
        return {"best_experiment": "", "best_metrics": {}, "ranking": [], "summary": "No results to rank."}

    scored = []
    for r in results:
        metrics = r.get("metrics", {})
        scored.append({
            "experiment_name": r.get("experiment_name", "unknown"),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "f1": metrics.get("f1", 0.0),
            "recall": metrics.get("recall", 0.0),
            "precision": metrics.get("precision", 0.0),
            "avg_precision": metrics.get("avg_precision", 0.0),
        })

    scored.sort(key=lambda x: (x["roc_auc"], x["f1"]), reverse=True)

    ranking = []
    for i, r in enumerate(scored):
        ranking.append({
            "rank": i + 1,
            "experiment_name": r["experiment_name"],
            "roc_auc": r["roc_auc"],
            "f1": r["f1"],
            "recall": r["recall"],
            "precision": r["precision"],
        })

    best = scored[0]
    worst = scored[-1]
    gain = round(best["roc_auc"] - worst["roc_auc"], 4)

    summary = (
        f"Ran {len(results)} experiments. Best: {best['experiment_name']} "
        f"(ROC-AUC={best['roc_auc']}, F1={best['f1']}, Recall={best['recall']}). "
        f"Worst: {worst['experiment_name']} (ROC-AUC={worst['roc_auc']}). "
        f"Gap between best and worst: {gain} ROC-AUC."
    )

    return {
        "best_experiment": best["experiment_name"],
        "best_metrics": {
            "roc_auc": best["roc_auc"],
            "f1": best["f1"],
            "recall": best["recall"],
            "precision": best["precision"],
        },
        "ranking": ranking,
        "summary": summary,
    }
