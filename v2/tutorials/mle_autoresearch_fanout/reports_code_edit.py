"""Agent run helpers for the code-edit autoresearch variant."""

from __future__ import annotations

from typing import Any

from autoresearch_types import DEFAULT_MAX_STEPS, DatasetProfile
from research_history import format_research_history_for_directive


def directive_code_edit(
    n_experiments: int,
    profile: DatasetProfile,
    memory_key: str,
    *,
    history: dict[str, Any] | None = None,
) -> str:
    """Build the user directive for the code-edit MLE agent."""
    history_block = format_research_history_for_directive(history or {})

    return (
        f"Run {n_experiments} code-edit experiments to minimize val_bpb on the climbmix corpus "
        f"({profile.n_parquet_files} parquet shards, vocab_size={profile.vocab_size}). "
        f"Use memory_key={memory_key!r} for edit_train_code, read_train_code, get_code_edit_history, "
        f"get_promising_code, record_hypothesis, get_leaderboard, and compare_experiments. "
        f"Each experiment: get_baseline_train_code or read_train_code → edit_train_code → "
        f"record_hypothesis → run_experiment (runs your edited train.py in a sandbox). "
        f"Use time_budget_sec=45, max_steps={DEFAULT_MAX_STEPS}. The platform right-sizes compute "
        f"and retries on sandbox OOM (inspect stderr on failure).\n\n"
        "Make each experiment explore a **different idea**: vary architecture "
        "(n_layer, n_head, n_embd), optimization (learning_rate, device_batch_size), and "
        "regularization (dropout) — one or two knobs per run, with a clear hypothesis. "
        "Do not repeat the same train.py under a new title.\n\n"
        "If this memory key already has prior trials, start with ``get_code_edit_history()`` "
        "and ``read_train_code`` on the current best title — do not re-run listed experiments."
        f"{history_block}"
    )
