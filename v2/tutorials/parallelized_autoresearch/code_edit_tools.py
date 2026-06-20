"""Agent tools for editing and recalling ``train.py`` in memory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flyte.ai.agents import MemoryStore, tool

from autoresearch_types import (
    CONFIG_ONLY_EDIT_LIMIT,
    ExperimentConfig,
    MAX_DEVICE_BATCH_SIZE,
    MAX_MAX_STEPS,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
)
from bundle import agent_env

MEMORY_KEY = "mle-autoresearch-code"

_CONFIG_FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)} - {"title"}
_RUN_TRAINING_DOC = re.compile(
    r"(def run_training\(config: ExperimentConfig\)[^:]*:\n(?:    \"\"\"[\s\S]*?\"\"\"\n))"
)


def normalize_train_py(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def baseline_train_py() -> str:
    """Return the repo baseline ``train.py`` (single source of truth for diffs)."""
    import train

    assert train.__file__ is not None
    return Path(train.__file__).read_text()


def filter_config_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    filtered = {k: v for k, v in overrides.items() if k in _CONFIG_FIELDS}
    if "n_layer" in filtered:
        filtered["n_layer"] = max(1, min(int(filtered["n_layer"]), MAX_N_LAYER))
    if "n_head" in filtered:
        filtered["n_head"] = max(1, min(int(filtered["n_head"]), MAX_N_HEAD))
    if "n_embd" in filtered:
        filtered["n_embd"] = max(1, min(int(filtered["n_embd"]), MAX_N_EMBD))
    if "device_batch_size" in filtered:
        filtered["device_batch_size"] = max(1, min(int(filtered["device_batch_size"]), MAX_DEVICE_BATCH_SIZE))
    if "max_steps" in filtered:
        filtered["max_steps"] = max(1, min(int(filtered["max_steps"]), MAX_MAX_STEPS))
    if "n_embd" in filtered and "n_head" in filtered and int(filtered["n_embd"]) % int(filtered["n_head"]) != 0:
        head = int(filtered["n_head"])
        filtered["n_embd"] = (int(filtered["n_embd"]) // head) * head
    return filtered


def is_config_only_edit(train_py: str, overrides: dict[str, Any] | None) -> bool:
    """True when *train_py* differs from baseline only via ``config_overrides`` injection."""
    baseline = baseline_train_py()
    filtered = filter_config_overrides(overrides)
    if not filtered:
        return normalize_train_py(train_py) == normalize_train_py(baseline)
    expected = build_train_py_with_config_overrides(baseline, filtered)
    return normalize_train_py(train_py) == normalize_train_py(expected)


def experiment_config_signature(train_py: str, overrides: dict[str, Any] | None) -> str:
    """Stable hash of effective train code + config overrides for duplicate detection."""
    filtered = filter_config_overrides(overrides)
    payload = {
        "train_py": normalize_train_py(train_py),
        "overrides": sorted(filtered.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


async def check_duplicate_config(
    memory_key: str,
    title: str,
    train_py: str,
    overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return duplicate metadata if this config was already run under another title."""
    sig = experiment_config_signature(train_py, overrides)
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    sigs: dict[str, str] = await memory.read_json.aio("memory/config_signatures.json", default={})
    prior_title = sigs.get(sig)
    title_key = title.strip().lower()
    if prior_title and prior_title.strip().lower() != title_key:
        return {"duplicate_of": prior_title, "config_signature": sig}
    return None


async def register_config_signature(
    memory_key: str,
    title: str,
    train_py: str,
    overrides: dict[str, Any] | None,
    *,
    actor: str = "mle-autoresearch-code-agent",
) -> str:
    """Record the config signature for *title* after a successful edit or run."""
    sig = experiment_config_signature(train_py, overrides)
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    sigs: dict[str, str] = await memory.read_json.aio("memory/config_signatures.json", default={})
    sigs[sig] = title
    await memory.write_json.aio(
        "memory/config_signatures.json",
        sigs,
        actor=actor,
        reason=f"config signature for {title}",
    )
    await memory.save.aio()
    return sig


def build_train_py_with_config_overrides(
    base_code: str,
    overrides: dict[str, Any],
) -> str:
    """Inject ``dataclasses.replace(config, ...)`` at the top of ``run_training``."""
    filtered = filter_config_overrides(overrides)
    if not filtered:
        return base_code

    parts = [f"{k}={v!r}" for k, v in sorted(filtered.items())]
    injection = f"    import dataclasses\n    config = dataclasses.replace(config, {', '.join(parts)})\n"
    match = _RUN_TRAINING_DOC.search(base_code)
    if match:
        insert_at = match.end()
        return base_code[:insert_at] + injection + base_code[insert_at:]
    return base_code


async def load_config_overrides(memory_key: str, title: str) -> dict[str, Any]:
    """Load persisted ``ExperimentConfig`` overrides for an experiment title."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    slug = slugify(title)
    stored = await memory.read_json.aio(f"memory/config/{slug}.json", default={})
    if stored:
        return filter_config_overrides(stored)

    index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    title_lower = title.strip().lower()
    for entry in index:
        if str(entry.get("title", "")).strip().lower() == title_lower:
            slug = str(entry.get("slug", slug))
            stored = await memory.read_json.aio(f"memory/config/{slug}.json", default={})
            if stored:
                return filter_config_overrides(stored)
            return filter_config_overrides(entry.get("config_overrides") or {})
    return {}


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:80] or "experiment"


async def load_train_code(memory_key: str, title: str) -> str:
    """Load edited ``train.py`` for *title*, falling back to the repo baseline."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    slug = slugify(title)
    saved = await memory.read_text.aio(f"memory/code/{slug}.py", default="")
    if saved.strip():
        return saved

    index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    title_lower = title.strip().lower()
    for entry in index:
        if str(entry.get("title", "")).strip().lower() == title_lower:
            slug = entry.get("slug", slug)
            saved = await memory.read_text.aio(f"memory/code/{slug}.py", default="")
            if saved.strip():
                return saved

    return baseline_train_py()


async def _global_best_val_bpb(memory: MemoryStore, *, exclude_title: str | None = None) -> float:
    """Lowest val_bpb recorded in memory (optionally excluding one title)."""
    exclude = (exclude_title or "").strip().lower()
    leaderboard: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    vals: list[float] = []
    for row in leaderboard + promising:
        if exclude and str(row.get("title", "")).strip().lower() == exclude:
            continue
        val = row.get("val_bpb")
        if val is not None:
            vals.append(float(val))
    return min(vals, default=float("inf"))


async def _update_promising_code(
    memory_key: str,
    *,
    title: str,
    slug: str,
    val_bpb: float,
    change_summary: str,
) -> None:
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    prior_best = await _global_best_val_bpb(memory, exclude_title=title)
    kept = val_bpb < prior_best
    promising.append(
        {
            "title": title,
            "slug": slug,
            "val_bpb": val_bpb,
            "kept": kept,
            "change_summary": change_summary,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    await memory.write_json.aio(
        "memory/promising_code.json",
        promising,
        actor="mle-autoresearch-code-agent",
        reason=f"promising code after {title} val_bpb={val_bpb}",
    )
    await memory.save.aio()


async def _resolve_train_py_for_edit(
    memory_key: str,
    spec: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    """Build the effective ``train.py`` source and overrides for one edit spec."""
    train_py = spec.get("train_py", "")
    if not isinstance(train_py, str):
        train_py = ""
    config_overrides = filter_config_overrides(
        spec.get("config_overrides") or spec.get("config") or {}
    )
    parent_title = spec.get("parent_title")
    parent_title = str(parent_title).strip() if parent_title else None

    baseline = baseline_train_py()
    if config_overrides:
        base_code = await load_train_code(memory_key, parent_title) if parent_title else baseline
        if not train_py.strip() or normalize_train_py(train_py) == normalize_train_py(baseline):
            train_py = build_train_py_with_config_overrides(base_code, config_overrides)
        elif parent_title and normalize_train_py(train_py) == normalize_train_py(base_code):
            train_py = build_train_py_with_config_overrides(base_code, config_overrides)

    return train_py, config_overrides, parent_title


async def _persist_train_edits(
    memory_key: str,
    edits: list[dict[str, Any]],
    *,
    actor: str = "mle-autoresearch-code-agent",
) -> dict[str, Any]:
    """Save one or more ``train.py`` edits in a single memory transaction."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    saved: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for spec in edits:
        title = str(spec.get("title", "")).strip()
        change_summary = str(spec.get("change_summary", ""))
        if not title:
            errors.append({"title": title or "(missing)", "saved": False, "error": "title is required"})
            continue

        train_py, config_overrides, parent_title = await _resolve_train_py_for_edit(memory_key, spec)
        if not train_py.strip():
            errors.append(
                {
                    "title": title,
                    "saved": False,
                    "error": "train_py or config_overrides is required",
                }
            )
            continue
        if is_config_only_edit(train_py, config_overrides) and len(index) >= CONFIG_ONLY_EDIT_LIMIT:
            errors.append(
                {
                    "title": title,
                    "saved": False,
                    "error": (
                        f"Batch 2+ requires substantive train.py edits (LR schedule, optimizer, "
                        f"weight decay, grad clip, etc.), not config_overrides alone. "
                        f"You already have {len(index)} saved edit(s)."
                    ),
                }
            )
            continue
        if normalize_train_py(train_py) == normalize_train_py(baseline_train_py()) and not config_overrides:
            errors.append(
                {
                    "title": title,
                    "saved": False,
                    "error": (
                        "train.py matches baseline with no config_overrides; "
                        "pass config_overrides={n_layer: 6, ...} or edit run_training"
                    ),
                }
            )
            continue
        if "def run_training" not in train_py:
            errors.append(
                {
                    "title": title,
                    "saved": False,
                    "error": "train_py must define run_training(config) like the baseline train.py",
                }
            )
            continue

        slug = slugify(title)
        await memory.write_text.aio(
            f"memory/code/{slug}.py",
            train_py,
            actor=actor,
            reason=f"edit train.py for {title}",
        )
        if config_overrides:
            await memory.write_json.aio(
                f"memory/config/{slug}.json",
                config_overrides,
                actor=actor,
                reason=f"config overrides for {title}",
            )
        index.append(
            {
                "title": title,
                "slug": slug,
                "change_summary": change_summary,
                "lines": len(train_py.splitlines()),
                "edited_at": now,
                "config_overrides": config_overrides,
                "parent_title": parent_title,
            }
        )
        saved.append(
            {
                "saved": True,
                "title": title,
                "slug": slug,
                "lines": len(train_py.splitlines()),
                "change_summary": change_summary,
                "train_py": train_py,
                "config_overrides": config_overrides,
                "parent_title": parent_title,
                "memory_path": f"memory/code/{slug}.py",
            }
        )

    if saved:
        await memory.write_json.aio(
            "memory/code_index.json",
            index,
            actor=actor,
            reason=f"code index update ({len(saved)} edit(s))",
        )
        await memory.save.aio()

    return {
        "count": len(saved),
        "titles": [row["title"] for row in saved],
        "edits": saved,
        "errors": errors,
    }


@tool
@agent_env.task
async def get_baseline_train_code() -> dict:
    """Return the baseline ``train.py`` from the repo (the karpathy/autoresearch recipe).

    Use this once at the start to understand the starting point before editing.

    Returns:
        A dict with ``title``, ``train_py`` (full source), and ``lines``.
    """
    code = baseline_train_py()
    return {"title": "baseline", "train_py": code, "lines": len(code.splitlines())}


@tool
@agent_env.task
async def edit_train_code(
    title: str,
    train_py: str = "",
    change_summary: str = "",
    memory_key: str = MEMORY_KEY,
    config_overrides: dict[str, Any] | None = None,
    parent_title: str | None = None,
) -> dict:
    """Save an edited ``train.py`` for this experiment to agent memory.

    The code must keep a ``run_training(config: ExperimentConfig) -> ExperimentResult``
    entry point (same as the baseline). Only edit architecture, optimizer, and
    training-loop knobs inside the file.

    Alternatively pass ``config_overrides`` (e.g. ``{"n_layer": 6, "learning_rate": 1e-4}``)
    instead of a full ``train_py`` rewrite — the platform injects
    ``dataclasses.replace(config, ...)`` into ``run_training`` for you.

    Args:
        title: Short human-readable experiment name (used as the memory key slug).
        train_py: Full Python source for the edited training script (optional if
            ``config_overrides`` is set).
        change_summary: One-line description of what you changed and why.
        memory_key: Memory namespace from your directive.
        config_overrides: Optional ``ExperimentConfig`` field overrides.
        parent_title: Optional prior experiment to fork before applying overrides.

    Returns:
        Metadata about the saved edit, including the full ``train_py`` source
        (visible in the Flyte task output UI).
    """
    result = await _persist_train_edits(
        memory_key,
        [
            {
                "title": title,
                "train_py": train_py,
                "change_summary": change_summary,
                "config_overrides": config_overrides,
                "parent_title": parent_title,
            }
        ],
    )
    if result["edits"]:
        return result["edits"][0]
    err = result["errors"][0] if result["errors"] else {"saved": False, "error": "unknown error"}
    return err


@tool
@agent_env.task
async def edit_train_code_batch(
    edits: list[dict[str, Any]],
    memory_key: str = MEMORY_KEY,
) -> dict:
    """Save multiple edited ``train.py`` files in one atomic memory write.

    Use this when preparing a parallel experiment batch — avoids sequential
    ``edit_train_code`` calls and race conditions on ``memory/code_index.json``.

    Each item in ``edits`` must include ``title`` and ``change_summary``, plus either
    ``train_py`` (full source) or ``config_overrides`` (e.g. ``{"n_layer": 6}``).
    Optional ``parent_title`` forks a prior experiment before applying overrides.
    Every ``train_py`` must keep the ``run_training(config)`` entry point.

    Args:
        edits: List of edit specs, e.g.
            ``[{"title": "deeper-6L", "config_overrides": {"n_layer": 6}, "change_summary": "..."}]``.
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``count``, ``titles``, ``edits`` (each includes ``train_py``),
        and ``errors`` (rejected).
    """
    if not edits:
        return {"count": 0, "titles": [], "edits": [], "errors": [{"error": "edits list is empty"}]}
    return await _persist_train_edits(
        memory_key,
        edits,
        actor="mle-autoresearch-code-fanout-agent",
    )


@tool
@agent_env.task
async def read_train_code(title: str, memory_key: str = MEMORY_KEY) -> dict:
    """Read a previously saved ``train.py`` edit from memory (or the baseline).

    Args:
        title: Experiment title whose code you want to inspect.
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``title``, ``train_py``, and ``lines``.
    """
    code = await load_train_code(memory_key, title)
    return {"title": title, "train_py": code, "lines": len(code.splitlines())}


@tool
@agent_env.task
async def get_promising_code(memory_key: str = MEMORY_KEY) -> dict:
    """Return promising ``train.py`` edits, the current best, and deltas vs best.

    Each entry records ``val_bpb`` after a successful run. Use ``read_train_code``
    with the best entry's title to inspect its source. Prefer ``get_code_edit_history``
    for the full cross-session table of edits, results, and regressions.

    Args:
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``entries``, ``best``, ``best_val_bpb``, and ``count``.
    """
    from research_history import load_research_history

    history = await load_research_history(memory_key)
    best_val = history.get("best_val_bpb")
    entries: list[dict[str, Any]] = []
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    for row in promising:
        val = row.get("val_bpb")
        val_f = float(val) if val is not None else None
        entries.append(
            {
                **row,
                "beat_best": val_f is not None and best_val is not None and val_f <= best_val,
                "delta_vs_best": (val_f - best_val) if val_f is not None and best_val is not None else None,
            }
        )
    best: dict[str, Any] | None = None
    if history.get("best_title"):
        best_key = str(history["best_title"]).strip().lower()
        for entry in reversed(entries):
            if str(entry.get("title", "")).strip().lower() == best_key:
                best = entry
                break
    return {
        "entries": entries,
        "best": best,
        "best_val_bpb": best_val,
        "best_title": history.get("best_title"),
        "count": len(entries),
    }


@tool
@agent_env.task
async def get_code_edit_history(memory_key: str = MEMORY_KEY) -> dict:
    """Return all prior code edits, run results, and whether each beat the current best.

    Call this at the start of a session when ``memory_key`` already has experiments.
    Shows every saved ``train.py`` edit, its ``change_summary``, ``val_bpb`` (if run),
    ``delta_vs_best`` (negative means better), ``outcome`` (``new_best`` / ``regression`` /
    ``failed`` / ``not_run``), and linked hypotheses.

    Args:
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``best_val_bpb``, ``best_title``, ``trials``, and summary counts.
    """
    from research_history import load_research_history

    return await load_research_history(memory_key)


async def load_saved_code_edits(memory_key: str) -> list[dict[str, Any]]:
    """Load all saved ``train.py`` edits from memory for reporting."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    val_by_title = {
        str(row.get("title", "")).strip().lower(): row.get("val_bpb")
        for row in promising
        if row.get("val_bpb") is not None
    }
    kept_titles = {
        str(row.get("title", "")).strip().lower()
        for row in promising
        if row.get("kept")
    }

    baseline = baseline_train_py()
    edits: list[dict[str, Any]] = []
    for entry in index:
        slug = str(entry.get("slug", slugify(str(entry.get("title", "")))))
        train_py = await memory.read_text.aio(f"memory/code/{slug}.py", default="")
        title = str(entry.get("title", ""))
        title_key = title.strip().lower()
        config_overrides = filter_config_overrides(entry.get("config_overrides") or {})
        if not config_overrides:
            config_overrides = filter_config_overrides(
                await memory.read_json.aio(f"memory/config/{slug}.json", default={})
            )
        if config_overrides and normalize_train_py(train_py) == normalize_train_py(baseline):
            parent_title = entry.get("parent_title")
            base_code = (
                await load_train_code(memory_key, str(parent_title))
                if parent_title
                else baseline
            )
            train_py = build_train_py_with_config_overrides(base_code, config_overrides)
        edits.append(
            {
                **entry,
                "slug": slug,
                "train_py": train_py,
                "config_overrides": config_overrides,
                "memory_path": f"memory/code/{slug}.py",
                "val_bpb": val_by_title.get(title_key),
                "kept": title_key in kept_titles,
            }
        )
    return edits


async def record_experiment_result(
    memory_key: str,
    result: dict[str, Any],
    *,
    actor: str = "mle-autoresearch-code-agent",
) -> None:
    """Upsert one experiment outcome into ``memory/leaderboard.json``."""
    title = str(result.get("title", "")).strip()
    if not title:
        return
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    leaderboard: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    row: dict[str, Any] = {
        "title": title,
        "success": bool(result.get("success")),
        "val_bpb": float(result["val_bpb"]) if result.get("val_bpb") is not None else None,
        "model_name": result.get("model_name"),
        "n_params": result.get("n_params"),
        "steps": int(result["steps"]) if result.get("steps") is not None else None,
        "resources": result.get("resources"),
        "oom_retries": int(result.get("oom_retries", 0)),
    }
    if not result.get("success"):
        err = result.get("error") or result.get("stderr") or "failed"
        row["error"] = str(err)[:200]

    title_key = title.lower()
    replaced = False
    for idx, existing in enumerate(leaderboard):
        if str(existing.get("title", "")).strip().lower() == title_key:
            leaderboard[idx] = row
            replaced = True
            break
    if not replaced:
        leaderboard.append(row)

    await memory.write_json.aio(
        "memory/leaderboard.json",
        leaderboard,
        actor=actor,
        reason=f"experiment result for {title}",
    )
    await memory.save.aio()


async def record_promising_run(
    memory_key: str,
    title: str,
    result: dict[str, Any],
    change_summary: str = "",
) -> None:
    """Persist a successful run's code to the promising-code ledger."""
    if not result.get("success") or result.get("val_bpb") is None:
        return
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    code_index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    summary = change_summary
    slug = slugify(title)
    for entry in reversed(code_index):
        if str(entry.get("title", "")).strip().lower() == title.strip().lower():
            summary = summary or str(entry.get("change_summary", ""))
            slug = str(entry.get("slug", slug))
            break
    await _update_promising_code(
        memory_key,
        title=title,
        slug=slug,
        val_bpb=float(result["val_bpb"]),
        change_summary=summary or "successful run",
    )
