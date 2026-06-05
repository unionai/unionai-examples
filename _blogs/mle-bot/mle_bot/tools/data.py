"""Data loading, profiling, and splitting tools.

These tools are safe, general-purpose, and side-effect free.
They run as durable Flyte tasks so they execute in the cloud on managed compute.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def profile_dataset(data: File, target_column: str) -> dict:
    """Profile a dataset and return statistics that inform ML problem design.

    Call this first before designing any experiments. The returned profile tells
    you the shape, column types, class balance, missing values, and numeric
    statistics — everything needed to choose algorithms and feature strategies.

    Args:
        data: CSV file to profile.
        target_column: Name of the column to predict.

    Returns a dict with keys:
        - shape: [n_rows, n_cols]
        - columns: list of all column names
        - dtypes: {col: dtype_string, ...}
        - numeric_columns: list of numeric column names (excluding target)
        - categorical_columns: list of non-numeric column names (excluding target)
        - target_distribution: {class_value: count, ...}
        - class_balance: {class_value: pct, ...}  (proportions, sum to 100)
        - missing_pct: {col: pct_missing, ...}
        - numeric_stats: {col: {mean, std, min, max, median}, ...}
        - n_classes: int — number of unique target values
        - is_imbalanced: bool — True if minority class < 20% of data
        - sample: list of 3 example rows as dicts
    """
    import numpy as np
    import pandas as pd

    path = await data.download()
    df = pd.read_csv(path)

    target_counts = df[target_column].value_counts()
    class_balance = (df[target_column].value_counts(normalize=True) * 100).round(2).to_dict()
    minority_pct = float(min(class_balance.values()))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_stats = {}
    for col in numeric_cols:
        if col == target_column:
            continue
        numeric_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "median": round(float(df[col].median()), 4),
        }

    # Point-biserial correlation between each numeric feature and the target
    feature_target_corr = {}
    for col in numeric_cols:
        if col == target_column:
            continue
        corr = float(df[col].corr(df[target_column]))
        if not np.isnan(corr):
            feature_target_corr[col] = round(corr, 4)
    # Sort by absolute correlation descending
    feature_target_corr = dict(
        sorted(feature_target_corr.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": [c for c in numeric_cols if c != target_column],
        "categorical_columns": [c for c in categorical_cols if c != target_column],
        "target_distribution": {str(k): int(v) for k, v in target_counts.items()},
        "class_balance": {str(k): float(v) for k, v in class_balance.items()},
        "missing_pct": {col: round(float(pct * 100), 2) for col, pct in df.isnull().mean().items()},
        "numeric_stats": numeric_stats,
        "feature_target_corr": feature_target_corr,
        "n_classes": int(df[target_column].nunique()),
        "is_imbalanced": minority_pct < 20.0,
        "sample": df.head(3).fillna("").to_dict(orient="records"),
    }


@tool_env.task
async def split_dataset(
    data: File,
    target_column: str,
    test_size: float,
    time_column: str,
    split_type: str,
) -> File:
    """Split a dataset and return either the train or test half.

    Call this twice — once with split_type="train" and once with split_type="test" —
    to get both halves. Always split before feature engineering to prevent data leakage.

    Args:
        data: CSV file to split.
        target_column: Name of the column to predict.
        test_size: Fraction of data to use for testing (e.g. 0.2 for 20%).
        time_column: If non-empty, sort by this column and take the last
                     `test_size` fraction as test (time-based split, no shuffling).
                     If empty string "", use stratified random split.
        split_type: Which half to return — "train" or "test".

    Returns:
        File — CSV file containing the requested split (train or test rows).
    """
    import tempfile

    import pandas as pd
    from flyte.io import File as FlyteFile
    from sklearn.model_selection import train_test_split

    path = await data.download()
    df = pd.read_csv(path)

    if time_column:
        df = df.sort_values(time_column).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[target_column],
            random_state=42,
        )

    selected_df = train_df if split_type == "train" else test_df

    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    selected_df.to_csv(out.name, index=False)
    out.close()
    return await FlyteFile.from_local(out.name)
