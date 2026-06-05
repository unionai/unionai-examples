"""Dataset resampling tool.

Addresses class imbalance at the data level — an alternative (or complement) to
algorithm-level class_weight / scale_pos_weight. Only ever resample the TRAIN split.
Applying resampling to the test split would give misleading evaluation metrics.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def resample_dataset(data: File, target_column: str, config: dict) -> File:
    """Resample a training dataset to address class imbalance.

    Use this on the TRAIN split only — never on test data. Resampling the test
    split would produce optimistic and misleading evaluation metrics.

    Supported strategies:
        "oversample":   Randomly duplicate minority class rows until the desired
                        ratio is reached. Fast, introduces no new information.
                        Good first try when the imbalance is severe.

        "undersample":  Randomly remove majority class rows. Loses information
                        but can dramatically speed up training and sometimes
                        improves recall by forcing the model to focus on the
                        minority class. Risky if majority class has important
                        boundary examples.

        "smote":        Synthetic Minority Over-sampling Technique. Generates new
                        synthetic minority samples by interpolating between existing
                        minority samples and their nearest neighbors. Produces more
                        diverse synthetic examples than random oversampling.
                        Requires at least k_neighbors + 1 minority samples.

    Args:
        data: CSV file to resample (should be the train split only).
        target_column: Name of the binary target column (values must be 0 and 1).
        config: Dict with keys:
            strategy (str): "oversample" | "undersample" | "smote". Required.
            target_ratio (float, default 0.2): Desired ratio of minority/majority
                after resampling. E.g., 0.2 means minority will be 20% of majority.
                Values > 1.0 are clamped to 1.0 (equal class sizes).
            k_neighbors (int, default 5): Number of nearest neighbors for SMOTE.
                Ignored for oversample/undersample.
            random_state (int, default 42): Random seed for reproducibility.

    Returns:
        File — resampled CSV with the same columns as the input.
    """
    import tempfile

    import numpy as np
    import pandas as pd
    from flyte.io import File as FlyteFile

    path = await data.download()
    df = pd.read_csv(path)

    strategy = config.get("strategy", "oversample")
    target_ratio = float(config.get("target_ratio", 0.2))
    k_neighbors = int(config.get("k_neighbors", 5))
    rng = np.random.default_rng(config.get("random_state", 42))

    minority_mask = df[target_column] == 1
    majority_mask = df[target_column] == 0
    minority_df = df[minority_mask]
    majority_df = df[majority_mask]

    n_majority = len(majority_df)
    n_minority = len(minority_df)
    target_minority = min(int(n_majority * target_ratio), n_majority)  # cap at 1:1

    if strategy == "oversample":
        if target_minority <= n_minority:
            resampled = df
        else:
            n_extra = target_minority - n_minority
            extra = minority_df.sample(n=n_extra, replace=True, random_state=int(rng.integers(1e6)))
            resampled = pd.concat([df, extra], ignore_index=True)

    elif strategy == "undersample":
        target_majority = int(n_minority / target_ratio) if target_ratio > 0 else n_majority
        target_majority = min(target_majority, n_majority)
        undersampled_majority = majority_df.sample(n=target_majority, replace=False,
                                                   random_state=int(rng.integers(1e6)))
        resampled = pd.concat([minority_df, undersampled_majority], ignore_index=True)

    elif strategy == "smote":
        from sklearn.neighbors import NearestNeighbors

        n_extra = max(0, target_minority - n_minority)
        if n_extra == 0 or n_minority < k_neighbors + 1:
            resampled = df
        else:
            feature_cols = [c for c in minority_df.columns
                            if c != target_column and pd.api.types.is_numeric_dtype(minority_df[c])]
            X_min = minority_df[feature_cols].values

            nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_min)))
            nn.fit(X_min)
            _, indices = nn.kneighbors(X_min)

            synthetic_rows = []
            for _ in range(n_extra):
                idx = int(rng.integers(len(X_min)))
                neighbor_idx = indices[idx, int(rng.integers(1, indices.shape[1]))]
                alpha = float(rng.uniform(0, 1))
                synthetic_features = X_min[idx] + alpha * (X_min[neighbor_idx] - X_min[idx])
                row = dict(zip(feature_cols, synthetic_features))
                row[target_column] = 1
                # Copy non-numeric columns from the base sample
                base_row = minority_df.iloc[idx]
                for col in minority_df.columns:
                    if col not in row:
                        row[col] = base_row[col]
                synthetic_rows.append(row)

            synthetic_df = pd.DataFrame(synthetic_rows, columns=df.columns)
            resampled = pd.concat([df, synthetic_df], ignore_index=True)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from: oversample, undersample, smote")

    # Shuffle so minority/majority rows are interleaved
    resampled = resampled.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)

    new_minority = int((resampled[target_column] == 1).sum())
    new_majority = int((resampled[target_column] == 0).sum())
    print(f"resample_dataset: {strategy} — {n_minority}/{n_majority} → {new_minority}/{new_majority} "
          f"(ratio {new_minority/new_majority:.3f})")

    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    resampled.to_csv(out.name, index=False)
    out.close()
    return await FlyteFile.from_local(out.name)
