"""Shared datatypes for the autoresearch MLE agent.

These dataclasses are the contract between the agent (which proposes
experiments), the training code (which runs them), and the report (which
visualizes them). Keeping them small and JSON-friendly means they serialize
cleanly as Flyte task inputs/outputs *and* as agent memory entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Workshop-sized bounds — keep CPU-only demo runs inside modest RAM.
DEFAULT_NUM_SHARDS = 1
MAX_N_LAYER = 6
MAX_N_EMBD = 256
MAX_N_HEAD = 8
MAX_DEVICE_BATCH_SIZE = 4
DEFAULT_MAX_STEPS = 100
MAX_MAX_STEPS = 500
# After this many saved edits, config_overrides-only changes are rejected (batch 2+).
CONFIG_ONLY_EDIT_LIMIT = 3


@dataclass
class ExperimentConfig:
    """One TinyGPT experiment the agent wants to try.

    These are the same knobs `karpathy/autoresearch` lets an agent edit in
    ``train.py`` — architecture (depth/width/heads), regularization, and
    optimization — surfaced as structured fields so the runtime can reason about
    the compute each config implies.
    """

    title: str = "baseline"
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    device_batch_size: int = 2
    learning_rate: float = 3e-4
    time_budget_sec: int = 45
    max_steps: int = DEFAULT_MAX_STEPS


@dataclass
class ExperimentResult:
    """The outcome of a single training run."""

    title: str
    val_bpb: float
    model_name: str
    n_params: int
    steps: int
    device: str
    config: ExperimentConfig
    notes: str = ""


@dataclass
class DatasetProfile:
    """A compact summary of the prepared bundle, shown to the agent."""

    n_parquet_files: int
    parquet_files: list[str]
    vocab_size: int
    data_bytes: int
    tokenizer_bytes: int


@dataclass
class HypothesisEntry:
    """A structured hypothesis recorded before an experiment."""

    title: str
    hypothesis: str
    expected_effect: str
    recorded_at: str = ""


@dataclass
class LeaderboardEntry:
    """A row in the experiment leaderboard parsed from the agent transcript."""

    index: int
    title: str
    val_bpb: float | None = None
    model_name: str | None = None
    n_params: int | None = None
    resources: str | None = None
    oom_retries: int = 0
    steps: int | None = None
    error: str | None = None
    kept: bool = False


@dataclass
class AutoresearchOutput:
    """Final output of the MLE autoresearch agent."""

    directive: str
    dataset_profile: DatasetProfile
    best: LeaderboardEntry | None
    leaderboard: list[LeaderboardEntry] = field(default_factory=list)
    summary: str = ""
    memory_key: str = ""
    total_experiments: int = 0
