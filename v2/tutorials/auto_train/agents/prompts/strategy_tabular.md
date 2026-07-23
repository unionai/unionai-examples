**Tabular strategy** (N={tab_n:,} samples, F={tab_f} features — {tab_num} numeric, {tab_cat} categorical):

Decision ladder:
1. **N < 10,000 → Gradient boosting** (LightGBM / XGBoost / CatBoost). Deep learning rarely beats GBMs at this scale.
2. **10,000 ≤ N < 100,000 → GBM first**, then try a small MLP (2–3 hidden layers) only if GBM has clearly plateaued.
3. **N ≥ 100,000 → GBM still competitive**; deep tabular models (TabNet, FT-Transformer, MLP with embeddings) become viable.
4. **Heavy categorical features** (high cardinality, >20 unique values): CatBoost (handles natively) or MLP with learned entity embeddings.
5. **Mostly numeric + low-cardinality categoricals** (<20 unique values): LightGBM / XGBoost with one-hot or ordinal encoding.

Feature engineering to consider:
- **Numeric**: log/sqrt transform for right-skewed features; polynomial interactions (degree 2) for small F; binning high-range features.
- **Categorical**: target encoding for high-cardinality (>20 unique); one-hot for low-cardinality.
- **Missing values**: median impute for numeric, mode/constant for categorical; add a binary missingness-indicator flag for columns with >5% missing.
- **Feature selection**: after the first GBM fit, drop features with zero importance.
- **Cross-validation**: for N < 10,000 prefer 5-fold stratified CV over a single 80/20 split.
