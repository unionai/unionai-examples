"""Pydantic schemas for tool inputs and agent data structures.

These models define the expected shape of configs and results throughout the agent.

Important: Tool functions that are called from the Monty sandbox must accept plain
`dict` at the boundary (Monty can't import or instantiate classes). Each tool parses
its incoming dict into the appropriate model internally for validation. In agent.py,
use `.model_dump()` to convert models back to dicts before passing to the sandbox.
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class FeatureConfig(BaseModel):
    """Configuration for the engineer_features tool."""

    group_column: str = Field(
        default="",
        description="Column to group by for rolling/lag features (e.g. 'machine_id'). "
                    "Required when rolling_columns or lag_columns is specified.",
    )
    time_column: str = Field(
        default="",
        description="Timestamp column to sort by before computing rolling/lag features.",
    )
    rolling_columns: list[str] = Field(
        default_factory=list,
        description="Numeric columns to compute rolling statistics for (mean, std, min, max).",
    )
    windows: list[int] = Field(
        default_factory=list,
        description="Rolling window sizes in rows (e.g. [6, 12, 24]).",
    )
    lag_columns: list[str] = Field(
        default_factory=list,
        description="Numeric columns to create lag features for.",
    )
    lags: list[int] = Field(
        default_factory=list,
        description="Lag steps in rows (e.g. [1, 3, 6]).",
    )
    normalize: bool = Field(
        default=False,
        description="If true, z-score normalize all numeric columns except target_column.",
    )
    target_column: str = Field(
        default="",
        description="Column to exclude from normalization. Required when normalize=True.",
    )
    drop_columns: list[str] = Field(
        default_factory=list,
        description="Columns to remove from output (e.g. raw timestamp after rolling).",
    )
    fillna_method: Literal["forward", "zero", "drop"] = Field(
        default="forward",
        description="How to fill NaN values introduced by rolling/lag. "
                    "'forward' forward-fills then fills remaining with 0. "
                    "'zero' fills all NaN with 0. 'drop' drops rows with NaN.",
    )


# ---------------------------------------------------------------------------
# Training hyperparameters (per algorithm)
# ---------------------------------------------------------------------------

class XGBoostParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=6, ge=1, le=20)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    scale_pos_weight: float = Field(
        default=1.0, ge=0,
        description="Set to n_negative/n_positive for imbalanced datasets.",
    )
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)


class RandomForestParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int | None = Field(
        default=None,
        description="Maximum tree depth. None means unlimited.",
    )
    min_samples_leaf: int = Field(default=1, ge=1)
    class_weight: Literal["balanced", "balanced_subsample"] | None = Field(default="balanced")


class GradientBoostingParams(BaseModel):
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    subsample: float = Field(default=1.0, gt=0, le=1)


class LogisticRegressionParams(BaseModel):
    C: float = Field(default=1.0, gt=0, description="Inverse regularization strength.")
    max_iter: int = Field(default=1000, ge=100)
    class_weight: Literal["balanced"] | None = Field(default="balanced")


# ---------------------------------------------------------------------------
# Experiment design (used by agent.py, validated when parsing LLM JSON)
# ---------------------------------------------------------------------------

Algorithm = Literal["xgboost", "random_forest", "gradient_boosting", "logistic_regression"]


class ExperimentConfig(BaseModel):
    """One experiment to run — produced by the LLM and executed by the agent."""

    name: str = Field(description="Short descriptive name for this experiment.")
    algorithm: Algorithm
    hyperparams: dict = Field(
        default_factory=dict,
        description="Algorithm-specific hyperparameters. Will be validated inside train_model.",
    )
    feature_config: FeatureConfig = Field(default_factory=FeatureConfig)
    rationale: str = Field(default="", description="Why this experiment is worth running.")


class InitialDesign(BaseModel):
    """LLM response for initial experiment design."""

    problem_type: str = Field(default="binary_classification")
    primary_metric: Literal["roc_auc", "f1", "recall"] = Field(default="roc_auc")
    reasoning: str
    experiments: list[ExperimentConfig]


class IterationDecision(BaseModel):
    """LLM response after analyzing experiment results."""

    should_continue: bool
    reasoning: str
    exploration_requests: list[dict] = Field(
        default_factory=list,
        description="Optional list of explore_dataset config dicts to run before designing "
                    "the next batch. Each dict is passed directly to explore_dataset.",
    )
    next_experiments: list[ExperimentConfig] = Field(default_factory=list)
