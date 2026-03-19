"""Model training tools.

A single unified interface for training classifiers with different algorithms.
The tool handles serialization, class imbalance, and basic hyperparameter passing.
"""

from flyte.io import File

from mle_bot.environments import tool_env
from mle_bot.schemas import (
    GradientBoostingParams,
    LogisticRegressionParams,
    RandomForestParams,
    XGBoostParams,
)


@tool_env.task
async def train_model(
    data: File,
    target_column: str,
    algorithm: str,
    hyperparams: dict,
) -> File:
    """Train a classification model and return the serialized model and training metrics.

    Supports multiple algorithms through a single interface so the agent can
    dispatch different approaches without knowing implementation details.

    Args:
        data: CSV file with training data (features + target column).
        target_column: Name of the column to predict.
        algorithm: One of:
            "xgboost"            — Gradient boosted trees. Good default for tabular data.
                                   Handles missing values and class imbalance natively.
            "random_forest"      — Ensemble of decision trees. More robust to outliers.
            "logistic_regression"— Linear model. Fast baseline, good for linearly separable problems.
            "gradient_boosting"  — Sklearn GradientBoostingClassifier. Slower than xgboost
                                   but sometimes better on small datasets.
        hyperparams: Dict of algorithm-specific hyperparameters. Common keys:
            For xgboost / gradient_boosting:
                n_estimators (int, default 100): Number of trees.
                max_depth (int, default 6): Maximum tree depth.
                learning_rate (float, default 0.1): Step size shrinkage.
                scale_pos_weight (float): Ratio negative/positive — use for imbalanced data.
                                          Set to (n_negative / n_positive) to upweight minority class.
                subsample (float, default 1.0): Fraction of samples used per tree.
                colsample_bytree (float, default 1.0): Fraction of features per tree.
            For random_forest:
                n_estimators (int, default 100): Number of trees.
                max_depth (int or null, default null): Maximum tree depth (null = unlimited).
                min_samples_leaf (int, default 1): Minimum samples at a leaf node.
                class_weight (str, default "balanced"): "balanced" reweights by class frequency.
            For logistic_regression:
                C (float, default 1.0): Inverse regularization strength (higher = less regularization).
                max_iter (int, default 1000): Maximum iterations for solver.
                class_weight (str, default "balanced"): "balanced" reweights by class frequency.

    Returns:
        File — serialized model (joblib format, contains model + feature columns + target column).
    """
    import tempfile

    import joblib
    import numpy as np
    import pandas as pd
    from flyte.io import File as FlyteFile
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    path = await data.download()
    df = pd.read_csv(path)

    # Only use numeric columns — drop strings like machine_id automatically
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    X = df[feature_cols].values
    y = df[target_column].values

    class_dist = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    default_scale = max(1.0, n_negative / n_positive) if n_positive > 0 else 1.0

    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        p = XGBoostParams.model_validate({**hyperparams, "scale_pos_weight": hyperparams.get("scale_pos_weight", default_scale)})
        params = {**p.model_dump(), "eval_metric": "logloss", "random_state": 42}
        model = XGBClassifier(**params)

    elif algorithm == "random_forest":
        p = RandomForestParams.model_validate(hyperparams)
        params = {**p.model_dump(), "random_state": 42, "n_jobs": -1}
        model = RandomForestClassifier(**params)

    elif algorithm == "gradient_boosting":
        p = GradientBoostingParams.model_validate(hyperparams)
        params = {**p.model_dump(), "random_state": 42}
        model = GradientBoostingClassifier(**params)

    elif algorithm == "logistic_regression":
        p = LogisticRegressionParams.model_validate(hyperparams)
        params = {**p.model_dump(), "random_state": 42}
        model = LogisticRegression(**params)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose from: xgboost, random_forest, gradient_boosting, logistic_regression")

    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    train_metrics = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred, average="binary", zero_division=0)), 4),
        "precision": round(float(precision_score(y, y_pred, average="binary", zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, average="binary", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_prob)), 4),
    }

    # Feature importance (top 20)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_dict = {feature_cols[i]: round(float(importances[i]), 4) for i in range(len(feature_cols))}
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
        importance_dict = {feature_cols[i]: round(float(importances[i]), 4) for i in range(len(feature_cols))}
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
    else:
        importance_dict = {}

    model_file = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump({"model": model, "feature_columns": feature_cols, "target_column": target_column}, model_file.name)
    model_file.close()

    return await FlyteFile.from_local(model_file.name)
