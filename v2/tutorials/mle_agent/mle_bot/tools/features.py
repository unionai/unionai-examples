"""Feature engineering tools.

General-purpose transformations that work on any tabular dataset.
Each tool takes a File and returns a File with the transformed data,
so they can be chained together in any order.
"""

from flyte.io import File

from mle_bot.environments import tool_env
from mle_bot.schemas import FeatureConfig


@tool_env.task
async def engineer_features(data: File, config: dict) -> File:
    """Apply feature engineering transformations to a dataset.

    This is the primary feature engineering tool. It applies a configurable
    set of transformations in a single pass and returns a new CSV file.

    Supported config keys (all optional):
        group_column (str): Column to group by for rolling features (e.g. "machine_id").
                            Required if rolling_columns is specified.
        time_column (str): Timestamp column to sort by before computing rolling features.
        rolling_columns (list[str]): Numeric column names to compute rolling statistics for.
        windows (list[int]): Rolling window sizes in rows (e.g. [6, 12, 24]).
                             Creates {col}_mean_{w}, {col}_std_{w}, {col}_min_{w}, {col}_max_{w}.
        lag_columns (list[str]): Numeric column names to create lag features for.
        lags (list[int]): Lag steps (e.g. [1, 3, 6]). Creates {col}_lag_{n}.
        normalize (bool): If true, z-score normalize all numeric columns (except target).
                          Apply this AFTER rolling/lag features. Default false.
        target_column (str): Column to exclude from normalization. Required if normalize=true.
        drop_columns (list[str]): Columns to remove from the output (e.g. raw timestamp).
        fillna_method (str): How to fill NaN values introduced by rolling/lag.
                             "zero" fills with 0, "forward" forward-fills, "drop" drops rows.
                             Default "forward".

    Example config:
        {
            "group_column": "machine_id",
            "time_column": "timestamp",
            "rolling_columns": ["vibration", "temperature", "pressure"],
            "windows": [6, 12, 24],
            "lag_columns": ["vibration"],
            "lags": [1, 3],
            "normalize": true,
            "target_column": "failure_24h",
            "drop_columns": ["timestamp"],
            "fillna_method": "forward"
        }

    Args:
        data: CSV file to transform.
        config: Dict describing which transformations to apply (see above).

    Returns:
        File — new CSV file with engineered features added (and optionally dropped columns removed).
        The original columns are preserved unless listed in drop_columns.
    """
    import tempfile

    import numpy as np
    import pandas as pd
    from flyte.io import File as FlyteFile

    cfg = FeatureConfig.model_validate(config)

    path = await data.download()
    df = pd.read_csv(path)

    group_col = cfg.group_column
    time_col = cfg.time_column
    rolling_cols = cfg.rolling_columns
    windows = cfg.windows
    lag_cols = cfg.lag_columns
    lags = cfg.lags
    normalize = cfg.normalize
    target_col = cfg.target_column
    drop_cols = cfg.drop_columns
    fillna_method = cfg.fillna_method

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    # Rolling features
    if rolling_cols and windows:
        if group_col and group_col in df.columns:
            for col in rolling_cols:
                if col not in df.columns:
                    continue
                for w in windows:
                    grouped = df.groupby(group_col)[col]
                    df[f"{col}_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
                    df[f"{col}_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
                    df[f"{col}_min_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).min())
                    df[f"{col}_max_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).max())
        else:
            for col in rolling_cols:
                if col not in df.columns:
                    continue
                for w in windows:
                    df[f"{col}_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
                    df[f"{col}_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
                    df[f"{col}_min_{w}"] = df[col].rolling(w, min_periods=1).min()
                    df[f"{col}_max_{w}"] = df[col].rolling(w, min_periods=1).max()

    # Lag features
    if lag_cols and lags:
        if group_col and group_col in df.columns:
            for col in lag_cols:
                if col not in df.columns:
                    continue
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(lag)
        else:
            for col in lag_cols:
                if col not in df.columns:
                    continue
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Fill NaN values
    if fillna_method == "zero":
        df = df.fillna(0)
    elif fillna_method == "forward":
        df = df.ffill().fillna(0)
    elif fillna_method == "drop":
        df = df.dropna().reset_index(drop=True)

    # Normalize numeric features
    if normalize:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [c for c in numeric_cols if c != target_col and c not in drop_cols]
        for col in cols_to_scale:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std

    # Drop requested columns
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(out.name, index=False)
    out.close()
    return await FlyteFile.from_local(out.name)
