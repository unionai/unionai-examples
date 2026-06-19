"""Batch orchestration tools for the fan-out code-edit MLE agent."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any

import flyte
from flyte.ai.agents import MemoryStore, tool

from autoresearch_types import HypothesisEntry
from bundle import agent_env

MEMORY_KEY_FANOUT = "mle-autoresearch-code-fanout"


@tool
@agent_env.task
async def record_batch_plan(
    batch_id: str,
    experiments: list[dict[str, Any]],
    memory_key: str = MEMORY_KEY_FANOUT,
) -> dict:
    """Persist a batch of planned experiments before editing or running them.

    Each experiment dict should include at least ``title`` and ``hypothesis``.
    Optional keys: ``expected_effect``, ``change_summary``, ``parent_title``.

    Args:
        batch_id: Short identifier for this batch (e.g. ``batch-1-depth-sweep``).
        experiments: Planned experiment specs for parallel execution.
        memory_key: Memory namespace from your directive.

    Returns:
        The saved batch record with ``batch_id``, ``count``, and ``experiments``.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    batches: list[dict[str, Any]] = await memory.read_json.aio("memory/batches.json", default=[])
    record = {
        "batch_id": batch_id,
        "experiments": experiments,
        "count": len(experiments),
        "status": "planned",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    batches.append(record)
    await memory.write_json.aio(
        "memory/batches.json",
        batches,
        actor="mle-autoresearch-code-fanout-agent",
        reason=f"batch plan {batch_id}",
    )
    await memory.save.aio()
    return record


@tool
@agent_env.task
async def get_batch_plan(batch_id: str, memory_key: str = MEMORY_KEY_FANOUT) -> dict:
    """Load a previously recorded batch plan by ``batch_id``.

    Args:
        batch_id: Identifier passed to ``record_batch_plan``.
        memory_key: Memory namespace from your directive.

    Returns:
        The batch record, or ``{"found": False}`` if missing.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    batches: list[dict[str, Any]] = await memory.read_json.aio("memory/batches.json", default=[])
    batch_id_lower = batch_id.strip().lower()
    for batch in reversed(batches):
        if str(batch.get("batch_id", "")).strip().lower() == batch_id_lower:
            return {"found": True, **batch}
    return {"found": False, "batch_id": batch_id}


@tool
@agent_env.task
async def record_batch_hypotheses(
    experiments: list[dict[str, Any]],
    memory_key: str = MEMORY_KEY_FANOUT,
) -> dict:
    """Record hypotheses for every experiment in a batch (before ``run_experiment_batch``).

    Each item needs ``title``, ``hypothesis``, and ``expected_effect``.

    Args:
        experiments: List of hypothesis dicts (one per planned experiment title).
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``recorded`` count and the appended entries.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    prior: list[dict[str, Any]] = await memory.read_json.aio("memory/hypotheses.json", default=[])
    recorded: list[dict[str, Any]] = []
    for spec in experiments:
        entry = HypothesisEntry(
            title=str(spec.get("title", "")),
            hypothesis=str(spec.get("hypothesis", "")),
            expected_effect=str(spec.get("expected_effect", "")),
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        row = dataclasses.asdict(entry)
        prior.append(row)
        recorded.append(row)
    await memory.write_json.aio(
        "memory/hypotheses.json",
        prior,
        actor="mle-autoresearch-code-fanout-agent",
        reason=f"batch hypotheses ({len(recorded)} experiments)",
    )
    await memory.save.aio()
    return {"recorded": len(recorded), "entries": recorded}


def evaluate_batch_results_impl(
    results: list[dict[str, Any]],
    batch_id: str = "",
) -> dict[str, Any]:
    """Rank and summarize the outcome of a parallel experiment batch."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            failures.append({"title": "?", "error": str(result)})
            continue
        if result.get("success") and result.get("val_bpb") is not None:
            successes.append(result)
        else:
            failures.append(
                {
                    "title": result.get("title", "?"),
                    "error": result.get("error") or (result.get("stderr") or "")[:200],
                    "oom": result.get("oom", False),
                }
            )

    ranked = sorted(successes, key=lambda r: float(r["val_bpb"]))
    best = ranked[0] if ranked else None
    return {
        "batch_id": batch_id or None,
        "total": len(results),
        "n_success": len(successes),
        "n_failed": len(failures),
        "ranked": [
            {
                "title": r.get("title"),
                "val_bpb": r.get("val_bpb"),
                "model_name": r.get("model_name"),
                "steps": r.get("steps"),
                "resources": r.get("resources"),
                "oom_retries": r.get("oom_retries", 0),
            }
            for r in ranked
        ],
        "best": best,
        "failures": failures,
    }


@tool
@agent_env.task
async def evaluate_batch_results(
    results: list[dict[str, Any]],
    batch_id: str = "",
) -> dict:
    """Rank and summarize the outcome of a parallel experiment batch.

    Use after ``run_experiment_batch`` or ``flyte_map("run_experiment", ...)``.
    Lower ``val_bpb`` is better.

    Args:
        results: List of ``run_experiment`` result dicts (same order as titles).
        batch_id: Optional batch label for the summary.

    Returns:
        A dict with ``successes``, ``failures``, ``ranked``, ``best``, and ``batch_id``.
    """
    return evaluate_batch_results_impl(results, batch_id=batch_id)


async def persist_run_results_to_leaderboard(
    memory_key: str,
    results: list[dict[str, Any]],
    *,
    actor: str = "mle-autoresearch-code-fanout-agent",
) -> int:
    """Persist run results (success or failure) to ``memory/leaderboard.json``."""
    from code_edit_tools import record_experiment_result

    added = 0
    for result in results:
        if not isinstance(result, dict) or not result.get("title"):
            continue
        await record_experiment_result(memory_key, result, actor=actor)
        added += 1
    return added


async def run_experiment_batch_impl(
    run_experiment_task: Any,
    titles: list[str],
    *,
    time_budget_sec: int = 45,
    memory_key: str = MEMORY_KEY_FANOUT,
    concurrency: int = 4,
    group_name: str | None = None,
) -> dict[str, Any]:
    """Fan out ``run_experiment`` across *titles* via ``flyte.map``."""
    if not titles:
        return {"batch_size": 0, "results": [], "titles": []}

    n = len(titles)
    budgets = [time_budget_sec] * n
    keys = [memory_key] * n
    map_kwargs: dict[str, Any] = {"concurrency": concurrency, "return_exceptions": True}
    if group_name:
        map_kwargs["group_name"] = group_name

    results: list[Any] = []
    async for item in flyte.map.aio(run_experiment_task, titles, budgets, keys, **map_kwargs):
        if isinstance(item, BaseException):
            results.append({"success": False, "title": "?", "error": str(item)})
        else:
            results.append(item)

    return {
        "batch_size": n,
        "titles": titles,
        "results": results,
        "concurrency": concurrency,
        "group_name": group_name,
    }
