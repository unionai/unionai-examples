"""Directive helpers for the fan-out code-edit MLE agent."""

from __future__ import annotations

from typing import Any

from autoresearch_types import DEFAULT_MAX_STEPS, DatasetProfile
from research_history import format_research_history_for_directive


def directive_code_edit_fanout(
    n_experiments: int,
    profile: DatasetProfile,
    memory_key: str,
    *,
    batch_size: int = 3,
    max_batches: int | None = None,
    history: dict[str, Any] | None = None,
) -> str:
    """Build the user directive for the code-mode fan-out agent."""
    if max_batches is None:
        max_batches = max(1, (n_experiments + batch_size - 1) // batch_size)

    history_block = format_research_history_for_directive(history or {})

    return (
        f"Run {n_experiments} code-edit experiments on climbmix "
        f"({profile.n_parquet_files} shards, vocab_size={profile.vocab_size}) using "
        f"**batched parallel fan-out**. Work in up to {max_batches} batch(es) of "
        f"{batch_size} hypotheses at a time.\n\n"
        f"Use memory_key={memory_key!r} for all memory-backed tools.\n\n"
        "Workflow (CODE MODE — write Python plans each turn):\n"
        "1. ``get_code_edit_history()`` (if prior trials exist) + ``get_baseline_train_code`` "
        "+ ``inspect_dataset``; optionally ``search_arxiv``.\n"
        "2. Plan a batch: ``record_batch_plan(batch_id, experiments=[...])``.\n"
        "3. **Batch 1:** ``edit_train_code_batch(edits=[...])`` may use ``config_overrides`` "
        "for architecture/LR sweeps. **Batch 2+:** each edit must include substantive "
        "``train_py`` changes (LR schedule, optimizer, weight decay, grad clip) — "
        "``config_overrides`` alone is rejected.\n"
        "4. ``record_batch_hypotheses([...])`` then ``run_experiment_batch(titles, ...)`` "
        f"OR ``flyte_map('run_experiment', titles, budgets, keys, concurrency={batch_size})``.\n"
        "5. ``evaluate_batch_results(results, batch_id=...)`` — pick the best, check ``steps`` "
        "in ranked results (deeper models should not starve for steps).\n"
        "6. Iterate: fork promising **train.py** edits into the next batch until "
        f"{n_experiments} experiments complete.\n"
        "7. Finish with a plain-text summary: best val_bpb, winning code changes, next batch idea.\n\n"
        f"**Batch diversity:** each parallel run must test a different hypothesis — spread "
        f"changes across training-loop code (batch 2+), depth/width, dropout, and batch size. "
        f"No duplicate configs (rejected at run time); no LR micro-sweeps within ±30% of best.\n\n"
        "**Plateau rule:** if 3 consecutive batches fail to beat the global best val_bpb by "
        ">0.01, stop hyperparameter sweeps and edit ``train.py`` (scheduler, optimizer, etc.).\n\n"
        "Do not repeat experiments already listed in prior research below. Fork the current "
        "best with ``read_train_code(best_title)`` before designing the next batch.\n\n"
        f"time_budget_sec=45, max_steps={DEFAULT_MAX_STEPS} (default). "
        f"time_budget is a safety cap; max_steps ensures fair comparison across architectures. "
        f"Platform retries sandbox OOM with more memory per run."
        f"{history_block}"
    )
