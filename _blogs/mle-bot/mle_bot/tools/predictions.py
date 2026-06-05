"""Prediction output tool.

Returns a model's predicted probabilities as a CSV file rather than just summary
metrics. This unlocks two advanced patterns only possible because of the sandbox's
ability to compose tools dynamically:

  1. Stacking: use one model's predictions as features for a second model.
  2. In-loop error analysis: call explore_dataset on the predictions file to
     see which samples are being misclassified, then use that insight to design
     the next experiment.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def get_predictions(model: File, data: File, target_column: str) -> File:
    """Run a trained model on data and return predictions as a CSV file.

    Unlike evaluate_model (which returns summary metrics), this returns a CSV
    with predicted probabilities and class labels for every row. Use this when
    you need to:

    - **Analyze errors**: pass the output to explore_dataset with
      class_distributions or group_stats to understand which kinds of examples
      the model gets wrong. For example: are failures at a particular machine
      or time window consistently misclassified?

    - **Stack models**: use the predicted_prob column as an input feature for a
      second model (meta-learner). Train a base model on the train split, get its
      predictions on the train split, then train a meta-model on those predictions
      + original features.

    - **Compare calibration**: look at the distribution of predicted_prob for
      true positives vs true negatives to understand whether the model's
      confidence is well-calibrated.

    Args:
        model: Serialized model file produced by train_model.
        data: CSV file to generate predictions for (can be train or test split).
        target_column: Name of the target column in data.

    Returns:
        File — CSV with all original columns plus:
            predicted_prob  (float) — probability of positive class
            predicted_class (int)   — predicted label at threshold 0.5
            correct         (int)   — 1 if prediction matches true label, else 0
    """
    import tempfile

    import joblib
    import pandas as pd
    from flyte.io import File as FlyteFile

    model_path = await model.download()
    data_path = await data.download()

    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_columns"]

    df = pd.read_csv(data_path)
    X = df[feature_cols].values
    y = df[target_column].values

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:, 1]
    else:
        probs = clf.predict(X).astype(float)

    preds = (probs >= 0.5).astype(int)
    correct = (preds == y).astype(int)

    out_df = df.copy()
    out_df["predicted_prob"] = probs.round(6)
    out_df["predicted_class"] = preds
    out_df["correct"] = correct

    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out_df.to_csv(out.name, index=False)
    out.close()
    return await FlyteFile.from_local(out.name)
