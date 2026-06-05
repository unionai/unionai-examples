"""Dataset exploration tool.

Flexible, targeted analysis that the agent can call at any point — after profiling,
after seeing model results, or from within the Monty sandbox during experiment planning.
Unlike profile_dataset (which gives a fixed summary once), explore_dataset answers
specific questions about the data as the agent's understanding evolves.
"""

from flyte.io import File

from mle_bot.environments import tool_env


@tool_env.task
async def explore_dataset(data: File, config: dict) -> dict:
    """Run targeted exploration queries against a dataset.

    Call this when you need specific information to guide a decision — for example,
    after seeing that vibration is a top feature, explore its class-conditional
    distribution; or after a model underperforms, look at temporal patterns to
    decide whether rolling features would help.

    Supported config keys (include one or more):

        class_distributions (list[str]):
            For each named column, return mean, std, min, max, and median
            separately for each target class. Reveals which features separate
            classes and by how much.

        correlation_matrix (bool):
            Full Pearson correlation matrix across all numeric features.
            Useful for spotting redundant features or unexpected relationships.

        temporal_trend (dict):
            {"time_column": str, "target_column": str, "freq": "1D" | "1h" | "6h" | "1W"}
            Aggregates target rate and feature means over time buckets.
            Use to detect drift, seasonality, or whether failure rate is stable.

        group_stats (dict):
            {"group_column": str, "target_column": str}
            Per-group failure rate, sample count, and feature means.
            Use when data has a natural grouping (e.g. machine_id, customer_id).

        outlier_summary (list[str]):
            For each named column, report the fraction of values beyond 3 std
            deviations and the 1st/99th percentiles. Helps decide whether
            normalization or clipping is needed.

        feature_target_corr_by_group (dict):
            {"group_column": str, "target_column": str, "feature_columns": list[str]}
            Per-group feature-target correlations. Reveals whether the signal
            is consistent across groups or varies machine-by-machine.

    Args:
        data: CSV file to explore (can be full dataset or a split).
        config: Dict of exploration queries (see above).

    Returns:
        dict with one key per requested analysis, containing the results.
    """
    import numpy as np
    import pandas as pd

    path = await data.download()
    df = pd.read_csv(path)
    results = {}

    # --- class_distributions ---
    if "class_distributions" in config:
        target_col = config.get("target_column", "")
        cols = config["class_distributions"]
        # Auto-detect target if not specified: first binary-ish numeric column
        if not target_col:
            for c in df.columns:
                if df[c].nunique() == 2:
                    target_col = c
                    break
        dist = {}
        if target_col and target_col in df.columns:
            for cls in sorted(df[target_col].unique()):
                subset = df[df[target_col] == cls]
                cls_stats = {}
                for col in cols:
                    if col not in df.columns:
                        continue
                    s = subset[col].dropna()
                    cls_stats[col] = {
                        "mean": round(float(s.mean()), 4),
                        "std": round(float(s.std()), 4),
                        "min": round(float(s.min()), 4),
                        "max": round(float(s.max()), 4),
                        "median": round(float(s.median()), 4),
                        "n": int(len(s)),
                    }
                dist[str(cls)] = cls_stats
        results["class_distributions"] = dist

    # --- correlation_matrix ---
    if config.get("correlation_matrix"):
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr().round(4)
        results["correlation_matrix"] = corr.to_dict()

    # --- temporal_trend ---
    if "temporal_trend" in config:
        tc = config["temporal_trend"]
        time_col = tc.get("time_column", "")
        target_col = tc.get("target_column", "")
        freq = tc.get("freq", "1D")
        # Normalize pandas 2.0+ renamed frequency aliases.
        # Extracts the numeric prefix and alphabetic suffix separately so that
        # e.g. "1M" → "1ME", "3Q" → "3QE", "6H" → "6h", "M" → "ME".
        import re as _re
        _freq_map = {
            "BYE": "BYE", "BQE": "BQE", "BME": "BME",
            "BY": "BYE", "BQ": "BQE", "BM": "BME",
            "YE": "YE", "QE": "QE", "ME": "ME",
            "Y": "YE", "Q": "QE", "M": "ME",
            "H": "h", "T": "min", "S": "s",
        }
        _m = _re.match(r'^(\d*)([A-Za-z]+)$', freq)
        if _m:
            _num, _alias = _m.groups()
            freq = _num + _freq_map.get(_alias, _alias)
        if time_col in df.columns and target_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df_t = df.set_index(time_col).sort_index()
            numeric_cols = df_t.select_dtypes(include=[np.number]).columns.tolist()
            agg = df_t[numeric_cols].resample(freq).agg(["mean", "sum", "count"])
            agg.columns = ["_".join(c) for c in agg.columns]
            agg.index = agg.index.astype(str)
            # Limit to 60 rows to keep the result manageable
            results["temporal_trend"] = agg.tail(60).to_dict()

    # --- group_stats ---
    if "group_stats" in config:
        gc = config["group_stats"]
        group_col = gc.get("group_column", "")
        target_col = gc.get("target_column", "")
        if group_col in df.columns and target_col in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            grp = df.groupby(group_col)[numeric_cols].agg(["mean", "count"]).round(4)
            grp.columns = ["_".join(c) for c in grp.columns]
            results["group_stats"] = grp.to_dict()

    # --- outlier_summary ---
    if "outlier_summary" in config:
        cols = config["outlier_summary"]
        outliers = {}
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            mean, std = s.mean(), s.std()
            outlier_frac = float(((s - mean).abs() > 3 * std).mean()) if std > 0 else 0.0
            outliers[col] = {
                "p1": round(float(s.quantile(0.01)), 4),
                "p99": round(float(s.quantile(0.99)), 4),
                "outlier_frac_3std": round(outlier_frac, 4),
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
            }
        results["outlier_summary"] = outliers

    # --- feature_target_corr_by_group ---
    if "feature_target_corr_by_group" in config:
        ftc = config["feature_target_corr_by_group"]
        group_col = ftc.get("group_column", "")
        target_col = ftc.get("target_column", "")
        feature_cols = ftc.get("feature_columns", [])
        if group_col in df.columns and target_col in df.columns:
            corr_by_group = {}
            for grp_val, grp_df in df.groupby(group_col):
                grp_corr = {}
                for col in feature_cols:
                    if col not in grp_df.columns:
                        continue
                    c = float(grp_df[col].corr(grp_df[target_col]))
                    if not np.isnan(c):
                        grp_corr[col] = round(c, 4)
                corr_by_group[str(grp_val)] = grp_corr
            results["feature_target_corr_by_group"] = corr_by_group

    return results
