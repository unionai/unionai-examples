"""Feature selection tool.

Reduces the feature set to the most informative columns. Most useful after
engineer_features creates many rolling/lag features — a dataset with 5 raw
sensors × 3 windows × 4 stats suddenly has 60+ features, many of which may
be redundant or add only noise.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def select_features(data: File, target_column: str, config: dict) -> File:
    """Select the most informative features from a dataset.

    Most useful after engineer_features when the feature count has grown large
    (20+ features). Reduces noise, speeds up training, and can improve
    generalization — especially for linear models and smaller datasets.

    Can be applied to the full dataset before splitting (minimal leakage risk
    for feature selection) or to train/test splits independently. When applied
    independently, the same features will typically be selected since the
    selection is based on stable statistical properties.

    Supported methods:
        "mutual_info":
            Ranks features by mutual information with the target and keeps the
            top k. Captures non-linear relationships. Best general-purpose
            method for classification. Requires target_column.

        "variance_threshold":
            Drops features whose variance is below a threshold. Removes
            near-constant features that carry no signal. Fast and safe — good
            as a first pass before mutual_info. Does not use the target.

        "correlation_filter":
            Drops one feature from each pair that is highly correlated with
            each other (redundancy reduction). Keeps the feature with higher
            correlation to the target. Useful when rolling features at
            different window sizes are all capturing the same trend.

    Args:
        data: CSV file to select features from.
        target_column: Name of the target column (preserved in output always).
        config: Dict with keys:
            method (str): "mutual_info" | "variance_threshold" | "correlation_filter".
            k (int): Number of features to keep. Required for mutual_info.
                     Ignored for other methods.
            threshold (float): Variance threshold (default 0.01) for
                               variance_threshold; correlation threshold
                               (default 0.95) for correlation_filter.
            keep_columns (list[str]): Columns to always keep regardless of
                                      selection (e.g., group IDs). Default [].

    Returns:
        File — CSV with selected features + target_column. Non-numeric columns
        listed in keep_columns are preserved. Other non-numeric columns are dropped.
    """
    import tempfile

    import numpy as np
    import pandas as pd
    from flyte.io import File as FlyteFile

    path = await data.download()
    df = pd.read_csv(path)

    method = config.get("method", "mutual_info")
    k = config.get("k", 20)
    threshold = config.get("threshold", None)
    keep_columns = config.get("keep_columns", [])

    # Candidate features: numeric columns excluding target and forced-keep columns
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_column and c not in keep_columns
    ]

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        X = df[numeric_cols].fillna(0).values
        y = df[target_column].values
        mi_scores = mutual_info_classif(X, y, random_state=42)
        scored = sorted(zip(numeric_cols, mi_scores), key=lambda x: x[1], reverse=True)
        selected = [col for col, _ in scored[:k]]
        print(f"select_features mutual_info: kept {len(selected)}/{len(numeric_cols)} features. "
              f"Top: {[c for c, _ in scored[:5]]}")

    elif method == "variance_threshold":
        from sklearn.feature_selection import VarianceThreshold
        thresh = threshold if threshold is not None else 0.01
        X = df[numeric_cols].fillna(0).values
        selector = VarianceThreshold(threshold=thresh)
        selector.fit(X)
        selected = [col for col, keep in zip(numeric_cols, selector.get_support()) if keep]
        dropped = len(numeric_cols) - len(selected)
        print(f"select_features variance_threshold={thresh}: dropped {dropped}, kept {len(selected)}")

    elif method == "correlation_filter":
        thresh = threshold if threshold is not None else 0.95
        corr_matrix = df[numeric_cols].corr().abs()
        # Compute feature-target correlation for tie-breaking
        target_corr = {col: abs(float(df[col].corr(df[target_column])))
                       for col in numeric_cols}
        to_drop = set()
        for i, col_a in enumerate(numeric_cols):
            if col_a in to_drop:
                continue
            for col_b in numeric_cols[i + 1:]:
                if col_b in to_drop:
                    continue
                if corr_matrix.loc[col_a, col_b] > thresh:
                    # Drop the one with lower target correlation
                    drop = col_b if target_corr.get(col_a, 0) >= target_corr.get(col_b, 0) else col_a
                    to_drop.add(drop)
        selected = [c for c in numeric_cols if c not in to_drop]
        print(f"select_features correlation_filter={thresh}: dropped {len(to_drop)}, kept {len(selected)}")

    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: mutual_info, variance_threshold, correlation_filter")

    output_cols = keep_columns + [target_column] + [c for c in selected if c not in keep_columns + [target_column]]
    output_cols = [c for c in output_cols if c in df.columns]
    out_df = df[output_cols]

    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out_df.to_csv(out.name, index=False)
    out.close()
    return await FlyteFile.from_local(out.name)
