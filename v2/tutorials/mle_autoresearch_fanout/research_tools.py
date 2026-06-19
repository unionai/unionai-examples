"""Lightweight research tools for the autoresearch MLE agent.

These complement :func:`mle_agent.run_experiment` with literature search, dataset
inspection, config validation, and memory-backed bookkeeping.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from flyte.ai.agents import MemoryStore, tool

from autoresearch_types import (
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_SHARDS,
    DatasetProfile,
    HypothesisEntry,
    MAX_DEVICE_BATCH_SIZE,
    MAX_MAX_STEPS,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
)
from bundle import agent_env, build_bundle, bundle_env, profile_bundle

MEMORY_KEY = "mle-autoresearch"


def estimate_n_params(n_layer: int, n_head: int, n_embd: int, vocab_size: int = 8192) -> int:
    """Rough parameter count for the TinyGPT architecture in ``train.py``."""
    import prepare

    _ = n_head  # head count affects layout, not total params for this estimate
    seq_len = prepare.MAX_SEQ_LEN
    params = vocab_size * n_embd + seq_len * n_embd  # token + position embeddings
    per_block = 12 * n_embd * n_embd + 4 * n_embd  # attn (qkv+proj) + mlp + layer norms
    params += n_layer * per_block
    params += 2 * n_embd + vocab_size * n_embd  # final ln + lm head
    return params


def _find_leaderboard_entry(entries: list[dict[str, Any]], title: str) -> dict[str, Any] | None:
    title_lower = title.strip().lower()
    for entry in entries:
        if str(entry.get("title", "")).strip().lower() == title_lower:
            return entry
    for entry in entries:
        if title_lower in str(entry.get("title", "")).strip().lower():
            return entry
    return None


@tool
@agent_env.task(retries=3)
async def search_arxiv(query: str, max_results: int = 4) -> str:
    """Search arXiv for recent papers relevant to the next experiment.

    Use this to gather external context on architectures, optimizers, or
    evaluation metrics before proposing a new TinyGPT configuration.

    Args:
        query: Free-text search query, e.g. ``small language model depth width``.
        max_results: Maximum number of papers to return (default 4).

    Returns:
        A markdown-ish bullet list of titles and short summaries, or a note
        if the search failed or returned nothing.
    """
    import httpx

    if not (query and query.strip()):
        return "(empty query; skip literature search)"

    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        lines: list[str] = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title_el = entry.find("atom:title", ns)
            title = " ".join((title_el.text or "").split())
            summary_el = entry.find("atom:summary", ns)
            summary = " ".join((summary_el.text or "").split())[:400]
            lines.append(f"- {title}\n  {summary}")
        return "\n".join(lines) if lines else "(no arXiv results; proceed without external context)"
    except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
        return f"(literature search failed: {exc})"
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code >= 500:
            return f"(literature search failed: {exc})"
        raise


@tool
@bundle_env.task(cache="auto")
async def inspect_dataset(num_shards: int = DEFAULT_NUM_SHARDS) -> dict:
    """Inspect the prepared climbmix corpus and BPE tokenizer bundle.

    Call this at the start of a research session to understand what data you
    are training on before spending experiment budget.

    Args:
        num_shards: Number of climbmix parquet shards to include in the bundle.

    Returns:
        A dict with shard/file metadata, vocab size, byte counts, and fixed
        training constants (``max_seq_len``, ``val_metric``).
    """
    import prepare

    bundle = await build_bundle(num_shards=num_shards)
    profile: DatasetProfile = await profile_bundle(bundle)
    return {
        **dataclasses.asdict(profile),
        "max_seq_len": prepare.MAX_SEQ_LEN,
        "val_metric": "val_bpb (lower is better)",
        "corpus": "karpathy/climbmix-400b-shuffle",
    }


@tool
@agent_env.task
async def validate_experiment_config(
    title: str,
    n_layer: int = 3,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.0,
    device_batch_size: int = 2,
    learning_rate: float = 3e-4,
    time_budget_sec: int = 45,
    max_steps: int = DEFAULT_MAX_STEPS,
    vocab_size: int = 8192,
) -> dict:
    """Validate a proposed experiment config before calling run_experiment.

    Checks structural constraints (e.g. ``n_embd`` divisible by ``n_head``)
    and returns an estimated parameter count so you can reason about model scale.

    Args:
        title: Short human-readable name for this experiment.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_embd: Embedding / hidden width.
        dropout: Dropout probability.
        device_batch_size: Sequences per training step.
        learning_rate: AdamW learning rate.
        time_budget_sec: Wall-clock training budget in seconds (safety cap).
        max_steps: Training steps before stopping (default {DEFAULT_MAX_STEPS}; fair across architectures).
        vocab_size: Vocabulary size (default matches the prepared tokenizer).

    Returns:
        A dict with ``valid``, ``errors``, ``estimated_n_params``, and ``title``.
    """
    errors: list[str] = []
    if n_layer < 1:
        errors.append("n_layer must be >= 1")
    if n_layer > MAX_N_LAYER:
        errors.append(f"n_layer must be <= {MAX_N_LAYER} (workshop limit)")
    if n_head < 1:
        errors.append("n_head must be >= 1")
    if n_head > MAX_N_HEAD:
        errors.append(f"n_head must be <= {MAX_N_HEAD} (workshop limit)")
    if n_embd < n_head:
        errors.append("n_embd must be >= n_head")
    if n_embd % n_head != 0:
        errors.append(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
    if n_embd > MAX_N_EMBD:
        errors.append(f"n_embd must be <= {MAX_N_EMBD} (workshop limit)")
    if device_batch_size < 1:
        errors.append("device_batch_size must be >= 1")
    if device_batch_size > MAX_DEVICE_BATCH_SIZE:
        errors.append(f"device_batch_size must be <= {MAX_DEVICE_BATCH_SIZE} (workshop limit)")
    if not (0.0 <= dropout <= 0.5):
        errors.append("dropout should be between 0.0 and 0.5")
    if learning_rate <= 0:
        errors.append("learning_rate must be > 0")
    if time_budget_sec < 10:
        errors.append("time_budget_sec should be at least 10 for meaningful training")
    if max_steps < 1:
        errors.append("max_steps must be >= 1")
    if max_steps > MAX_MAX_STEPS:
        errors.append(f"max_steps must be <= {MAX_MAX_STEPS} (workshop limit)")
    return {
        "title": title,
        "valid": not errors,
        "errors": errors,
        "estimated_n_params": estimate_n_params(n_layer, n_head, n_embd, vocab_size),
    }


@tool
@agent_env.task
async def record_hypothesis(
    title: str,
    hypothesis: str,
    expected_effect: str,
    memory_key: str = MEMORY_KEY,
) -> dict:
    """Record a structured hypothesis before running an experiment.

    Persists to the agent's keyed memory so later runs can see what you
    expected and whether it panned out.

    Args:
        title: Experiment title this hypothesis applies to.
        hypothesis: What you are trying and why.
        expected_effect: How you expect val_bpb to move (e.g. ``decrease ~5%``).
        memory_key: Memory namespace (use the key from your directive).

    Returns:
        The recorded hypothesis entry.
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    prior: list[dict[str, Any]] = await memory.read_json.aio("memory/hypotheses.json", default=[])
    entry = HypothesisEntry(
        title=title,
        hypothesis=hypothesis,
        expected_effect=expected_effect,
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    prior.append(dataclasses.asdict(entry))
    await memory.write_json.aio(
        "memory/hypotheses.json",
        prior,
        actor="mle-autoresearch-agent",
        reason=f"hypothesis for {title}",
    )
    await memory.save.aio()
    return dataclasses.asdict(entry)


@tool
@agent_env.task
async def get_leaderboard(memory_key: str = MEMORY_KEY) -> dict:
    """Return the persisted experiment leaderboard from agent memory.

    Use this to recall prior runs across sessions. Experiments from the
    *current* session also appear in your tool-call transcript.

    Args:
        memory_key: Memory namespace (use the key from your directive).

    Returns:
        A dict with ``entries`` (list) and ``best`` (entry or null).
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    entries: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    best: dict[str, Any] | None = None
    best_val = float("inf")
    for entry in entries:
        val = entry.get("val_bpb")
        if val is not None and float(val) < best_val:
            best_val = float(val)
            best = entry
    best_f = best_val if best_val != float("inf") else None
    enriched: list[dict[str, Any]] = []
    for entry in entries:
        val = entry.get("val_bpb")
        val_f = float(val) if val is not None else None
        enriched.append(
            {
                **entry,
                "beat_best": val_f is not None and best_f is not None and val_f <= best_f,
                "delta_vs_best": (val_f - best_f) if val_f is not None and best_f is not None else None,
            }
        )
    return {
        "entries": enriched,
        "best": best,
        "best_val_bpb": best_f,
        "count": len(enriched),
    }


@tool
@agent_env.task
async def compare_experiments(
    title_a: str,
    title_b: str,
    memory_key: str = MEMORY_KEY,
) -> dict:
    """Compare two prior experiments side-by-side.

    Looks up both titles in the persisted leaderboard. For experiments run in
    the current session that are not yet persisted, use the values from your
    recent ``run_experiment`` tool results instead.

    Args:
        title_a: Title of the first experiment.
        title_b: Title of the second experiment.
        memory_key: Memory namespace (use the key from your directive).

    Returns:
        A dict with ``a``, ``b``, and ``delta_val_bpb`` (a minus b; negative
        means a is better).
    """
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    entries: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    a = _find_leaderboard_entry(entries, title_a)
    b = _find_leaderboard_entry(entries, title_b)
    missing = [t for t, e in ((title_a, a), (title_b, b)) if e is None]
    delta: float | None = None
    if a is not None and b is not None and a.get("val_bpb") is not None and b.get("val_bpb") is not None:
        delta = float(a["val_bpb"]) - float(b["val_bpb"])
    return {
        "a": a,
        "b": b,
        "delta_val_bpb": delta,
        "missing": missing,
        "note": (
            "Some titles were not found in persisted memory; check recent run_experiment "
            "tool results in your transcript for the current session."
            if missing
            else None
        ),
    }
