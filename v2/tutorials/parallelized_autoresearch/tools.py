"""Agent tools, sandbox execution, and memory helpers for parallelized autoresearch."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flyte
import flyte.errors
from flyte.ai.agents import LLMCallable, LLMMessage, MemoryStore, ToolFn, tool
from flyte.ai.agents._llm import _default_call_llm

from autoresearch_types import (
    CONFIG_ONLY_EDIT_LIMIT,
    DEFAULT_NUM_SHARDS,
    DatasetProfile,
    ExperimentConfig,
    HypothesisEntry,
    MAX_DEVICE_BATCH_SIZE,
    MAX_MAX_STEPS,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
)
from bundle import agent_env, build_bundle, bundle_env, profile_bundle

MEMORY_KEY_FANOUT = "parallelized-autoresearch"



MAX_LLM_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0


async def call_llm(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> LLMMessage:
    """Call litellm via the Flyte default callback, retrying transient provider errors."""
    import litellm

    backoff = INITIAL_BACKOFF_SEC
    last_exc: Exception | None = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return await _default_call_llm(model, system, messages, tools)
        except litellm.InternalServerError as exc:
            last_exc = exc
            if attempt >= MAX_LLM_RETRIES - 1:
                break
            flyte.logger.warning(
                "LLM InternalServerError (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1,
                MAX_LLM_RETRIES,
                backoff,
                exc,
            )
            await asyncio.sleep(backoff)
            backoff *= 2
    assert last_exc is not None
    raise last_exc


RESOURCE_FLOOR = flyte.Resources(cpu=2, memory="2Gi")
RESOURCE_CEILING = flyte.Resources(cpu=16, memory="32Gi")
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$")


def _memory_to_mib(memory: str | None) -> int:
    if not memory:
        return 2048
    match = _MEM_RE.match(memory)
    if not match:
        return 2048
    value = float(match.group(1))
    unit = (match.group(2) or "Mi").lower()
    if unit in ("gi", "g", "gb"):
        return int(value * 1024)
    if unit in ("mi", "m", "mb"):
        return int(value)
    if unit in ("ki", "k", "kb"):
        return max(1, int(value // 1024))
    return int(value)


def _mib_to_memory(mib: int) -> str:
    if mib >= 1024 and mib % 1024 == 0:
        return f"{mib // 1024}Gi"
    return f"{mib}Mi"


def _cap_resources(resources: flyte.Resources) -> flyte.Resources:
    floor_cpu = int(RESOURCE_FLOOR.cpu or 2)
    ceil_cpu = int(RESOURCE_CEILING.cpu or 16)
    cpu = int(resources.cpu or floor_cpu)
    cpu = max(floor_cpu, min(ceil_cpu, cpu))

    floor_mib = _memory_to_mib(
        RESOURCE_FLOOR.memory if isinstance(RESOURCE_FLOOR.memory, str) else "2Gi"
    )
    ceil_mib = _memory_to_mib(
        RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
    )
    mem_mib = _memory_to_mib(resources.memory if isinstance(resources.memory, str) else None)
    mem_mib = max(floor_mib, min(ceil_mib, mem_mib))
    return flyte.Resources(cpu=cpu, memory=_mib_to_memory(mem_mib))


def _ensure_oom_increase(resources: flyte.Resources, previous: flyte.Resources) -> flyte.Resources:
    """If memory did not grow after OOM, bump deterministically up to the ceiling."""
    prev_mib = _memory_to_mib(previous.memory if isinstance(previous.memory, str) else None)
    new_mib = _memory_to_mib(resources.memory if isinstance(resources.memory, str) else None)
    if new_mib <= prev_mib:
        ceil_mib = _memory_to_mib(
            RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
        )
        new_mib = min(ceil_mib, max(prev_mib * 2, prev_mib + 2048))
        resources = dataclasses.replace(resources, memory=_mib_to_memory(new_mib))
    prev_cpu = int(previous.cpu or RESOURCE_FLOOR.cpu or 2)
    new_cpu = int(resources.cpu or prev_cpu)
    if new_cpu < prev_cpu:
        resources = dataclasses.replace(resources, cpu=prev_cpu)
    return _cap_resources(resources)


def bump_memory(resources: flyte.Resources) -> flyte.Resources:
    """Deterministic memory bump after OOM."""
    return _ensure_oom_increase(resources, resources)


MAX_OOM_RETRIES = 3

RESOURCE_SIZING_SYSTEM_PROMPT = """\
You are a Kubernetes capacity planner for Flyte autoresearch sandbox training runs. \
Given a task's name, its docstring, and the concrete arguments it is about to be \
called with, estimate the *minimum sensible* compute it needs to finish without \
being OOM-killed, while not wildly over-provisioning.

Reason about the work implied by the arguments:
- TinyGPT training is memory-bound: scale with model width/depth (n_layer, n_embd, \
n_head), device_batch_size, and sequence length (512 in this workshop).
- Larger models and batch sizes need more RAM; CPU helps dataloader throughput but \
memory is usually the bottleneck.
- Sandbox runs are capped at a short time_budget_sec wall clock — prefer enough \
memory to survive peak activation usage over extra CPU.

Respond with ONLY a JSON object (no prose, no code fences) with any of these keys:
  - "cpu":    a number of cores, e.g. 2, 4, 8
  - "memory": a Kubernetes memory string, e.g. "4Gi", "16Gi"
  - "disk":   a Kubernetes disk string, e.g. "10Gi" (omit unless large I/O)
Omit a key to accept the default. Do not include any other keys. No GPUs are \
available on this cluster.

Example response: {"cpu": 4, "memory": "8Gi"}
"""

_ALLOWED_RESOURCE_KEYS = ("cpu", "memory", "disk", "shm")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str | None) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from an LLM reply."""
    if not text:
        return {}
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _resources_from_spec(spec: dict[str, Any], floor: flyte.Resources) -> flyte.Resources:
    """Merge an LLM-produced spec onto the floor, keeping only known keys."""
    kwargs: dict[str, Any] = {
        "cpu": floor.cpu,
        "memory": floor.memory,
        "gpu": floor.gpu,
        "disk": floor.disk,
        "shm": floor.shm,
    }
    for key in _ALLOWED_RESOURCE_KEYS:
        value = spec.get(key)
        if value in (None, "", "null"):
            continue
        kwargs[key] = value
    try:
        return _cap_resources(flyte.Resources(**kwargs))
    except Exception as exc:  # pragma: no cover - defensive against bad model output
        flyte.logger.warning("Invalid resource spec %s (%s); falling back to floor.", spec, exc)
        return floor


async def estimate_resources(
    call_llm: LLMCallable,
    model: str,
    tool_name: str,
    description: str,
    args: dict[str, Any],
) -> flyte.Resources:
    """Ask the LLM to size the compute for a single tool call."""
    user = json.dumps({"tool": tool_name, "description": description, "arguments": args}, default=str)
    try:
        reply = await call_llm(
            model,
            RESOURCE_SIZING_SYSTEM_PROMPT,
            [{"role": "user", "content": user}],
            None,
        )
        spec = _extract_json(reply.content)
    except Exception as exc:  # pragma: no cover - never let sizing break the tool
        flyte.logger.warning("Resource right-sizing LLM call failed (%s); using floor.", exc)
        spec = {}
    resources = _resources_from_spec(spec, RESOURCE_FLOOR)
    flyte.logger.info("right-size %s %s -> %s", tool_name, args, resources)
    return resources


# {{docs-fragment right_size}}
async def execute_with_right_sizing(
    call_llm: LLMCallable,
    target_task: Any,
    *,
    model: str,
    tool_name: str,
    description: str,
    max_oom_retries: int = MAX_OOM_RETRIES,
    **kwargs: Any,
) -> dict:
    """LLM-size *target_task*, run it, and retry with more memory on OOM."""
    resources = await estimate_resources(call_llm, model, tool_name, description, kwargs)
    attempt = 0
    while True:
        try:
            with flyte.group(f"{tool_name}-attempt-{attempt + 1}"):
                result = await target_task.override(resources=resources).aio(**kwargs)
        except flyte.errors.OOMError:
            if attempt >= max_oom_retries:
                flyte.logger.error("%s Flyte OOM after %d retries; giving up.", tool_name, attempt)
                raise
            resources = bump_memory(resources)
            attempt += 1
            flyte.logger.warning(
                "%s Flyte OOM; retrying with memory=%s",
                tool_name,
                resources.memory,
            )
            continue

        if isinstance(result, dict):
            result["resources"] = f"cpu={resources.cpu}, mem={resources.memory}"
            result["oom_retries"] = attempt

        if isinstance(result, dict) and result.get("oom"):
            if attempt >= max_oom_retries:
                return result
            resources = bump_memory(resources)
            attempt += 1
            flyte.logger.warning(
                "%s sandbox OOM; retrying with memory=%s",
                tool_name,
                resources.memory,
            )
            continue

        return result


def right_sizing_handler(*, max_oom_retries: int = MAX_OOM_RETRIES):
    """Build a ``@tool`` ``call_handler`` that right-sizes and self-heals on OOM."""

    async def handle(call_llm: LLMCallable, tool_fn: ToolFn, **kwargs: Any) -> Any:
        return await execute_with_right_sizing(
            call_llm,
            tool_fn.target,
            model=tool_fn.model,
            tool_name=tool_fn.name,
            description=tool_fn.description,
            max_oom_retries=max_oom_retries,
            **kwargs,
        )

    return handle


right_size = right_sizing_handler(max_oom_retries=MAX_OOM_RETRIES)
# {{/docs-fragment right_size}}


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
async def record_hypothesis(
    title: str,
    hypothesis: str,
    expected_effect: str,
    memory_key: str = MEMORY_KEY_FANOUT,
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
async def get_leaderboard(memory_key: str = MEMORY_KEY_FANOUT) -> dict:
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
    memory_key: str = MEMORY_KEY_FANOUT,
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
    memory_key: str = MEMORY_KEY_FANOUT,
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
    memory_key: str = MEMORY_KEY_FANOUT,
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
        actor="parallelized-autoresearch",
    )


@tool
@agent_env.task
async def read_train_code(title: str, memory_key: str = MEMORY_KEY_FANOUT) -> dict:
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
async def get_promising_code(memory_key: str = MEMORY_KEY_FANOUT) -> dict:
    """Return promising ``train.py`` edits, the current best, and deltas vs best.

    Each entry records ``val_bpb`` after a successful run. Use ``read_train_code``
    with the best entry's title to inspect its source. Prefer ``get_code_edit_history``
    for the full cross-session table of edits, results, and regressions.

    Args:
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``entries``, ``best``, ``best_val_bpb``, and ``count``.
    """
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
async def get_code_edit_history(memory_key: str = MEMORY_KEY_FANOUT) -> dict:
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
        actor="parallelized-autoresearch",
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
        actor="parallelized-autoresearch",
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
    actor: str = "parallelized-autoresearch",
) -> int:
    """Persist run results (success or failure) to ``memory/leaderboard.json``."""
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


OOM_MARKERS = (
    "out of memory",
    "oom",
    "cannot allocate memory",
    "can't allocate memory",
    "unable to allocate",
    "memoryerror",
    "killed",
    "signal 9",
    "std::bad_alloc",
    "defaultcpuallocator",
    "bad_alloc",
)


def is_oom(stderr: str, returncode: int | None, *, stdout: str = "") -> bool:
    """Detect OOM from sandbox stderr / exit code (137 = SIGKILL/OOM-kill)."""
    if returncode in (137, -9):
        return True
    text = f"{stderr}\n{stdout}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def parse_metrics(stdout: str) -> dict[str, Any] | None:
    """Parse the ``AUTORESEARCH_METRICS=`` line emitted by the driver script."""
    for line in stdout.splitlines():
        if line.startswith("AUTORESEARCH_METRICS="):
            return json.loads(line.split("=", 1)[1])
    return None


def write_driver_script(title: str, time_budget_sec: int, eval_tokens: int) -> str:
    """Return a small driver that imports the agent-edited ``train.py`` and prints metrics."""
    return textwrap.dedent(
        f'''
        import json
        import os
        import sys

        workdir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(workdir)
        os.environ["AUTORESEARCH_CACHE"] = workdir
        sys.path.insert(0, workdir)
        os.environ.setdefault("AUTORESEARCH_EVAL_TOKENS", "{eval_tokens}")

        from autoresearch_types import ExperimentConfig
        import train

        overrides = {{}}
        overrides_path = os.path.join(workdir, "config_overrides.json")
        if os.path.exists(overrides_path):
            with open(overrides_path) as f:
                overrides = json.load(f)

        config = ExperimentConfig(title={title!r}, time_budget_sec={time_budget_sec})
        if overrides:
            import dataclasses
            config = dataclasses.replace(config, **overrides)
        result = train.run_training(config)
        payload = {{
            "title": result.title,
            "val_bpb": round(result.val_bpb, 6),
            "model_name": result.model_name,
            "n_params": result.n_params,
            "steps": result.steps,
            "device": result.device,
            "notes": result.notes,
        }}
        print("AUTORESEARCH_METRICS=" + json.dumps(payload))
        '''
    ).strip()


def stage_sandbox_files(
    work_dir: str,
    train_py: str,
    *,
    title: str,
    time_budget_sec: int,
    eval_tokens: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Copy support modules + edited train code into the sandbox work directory."""
    import autoresearch_types
    import prepare

    if eval_tokens is None:
        eval_tokens = 32 * prepare.MAX_SEQ_LEN
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.py").write_text(train_py)
    if config_overrides:
        (root / "config_overrides.json").write_text(json.dumps(config_overrides))
    (root / "prepare.py").write_text(Path(prepare.__file__).read_text())
    (root / "autoresearch_types.py").write_text(Path(autoresearch_types.__file__).read_text())
    driver = write_driver_script(title, time_budget_sec, eval_tokens)
    driver_path = root / "driver.py"
    driver_path.write_text(driver)
    return driver_path


async def run_train_in_sandbox(
    work_dir: str,
    train_py: str,
    *,
    title: str,
    time_budget_sec: int,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``train.py`` via ``async with sb.on_device.session(backend='userns')``."""
    from union import sandbox as sb

    driver_path = stage_sandbox_files(
        work_dir,
        train_py,
        title=title,
        time_budget_sec=time_budget_sec,
        config_overrides=config_overrides,
    )
    timeout_s = max(time_budget_sec + 180, 300)

    try:
        async with sb.on_device.session(backend="userns", host_work_dir=work_dir) as sbx:
            proc = await sbx.run(
                f"python {driver_path}",
                stdout=True,
                stderr=True,
                network_mode="blocked",
                timeout_s=timeout_s,
            )
            stdout, stderr = await proc.communicate_text()
    except Exception as exc:
        err_text = str(exc)
        oom = is_oom(err_text, None)
        return {
            "success": False,
            "oom": oom,
            "title": title,
            "exit_code": None,
            "stdout_tail": "",
            "stderr": err_text,
            "error": (
                "Training run was OOM-killed; the platform will retry with more memory."
                if oom
                else f"Sandbox execution failed: {err_text}"
            ),
        }

    metrics = parse_metrics(stdout or "")
    oom = is_oom(stderr or "", proc.returncode, stdout=stdout or "")

    if metrics is not None and proc.returncode == 0:
        return {
            "success": True,
            "oom": False,
            **metrics,
            "exit_code": proc.returncode,
            "stderr_tail": (stderr or "")[-800:],
        }

    return {
        "success": False,
        "oom": oom,
        "title": title,
        "exit_code": proc.returncode,
        "stdout_tail": (stdout or "")[-1500:],
        "stderr": stderr or "",
        "error": (
            "Training run was OOM-killed; the platform will retry with more memory."
            if oom
            else f"Training failed (exit {proc.returncode}). See stderr for details."
        ),
    }


def _title_key(title: str) -> str:
    return str(title or "").strip().lower()


def _best_from_entries(entries: list[dict[str, Any]]) -> tuple[float | None, str | None]:
    best_val: float | None = None
    best_title: str | None = None
    for row in entries:
        val = row.get("val_bpb")
        if val is None:
            continue
        fval = float(val)
        if best_val is None or fval < best_val:
            best_val = fval
            best_title = str(row.get("title", ""))
    return best_val, best_title


def _latest_by_title(rows: list[dict[str, Any]], *, title_field: str = "title") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _title_key(str(row.get(title_field, "")))
        if key:
            out[key] = row
    return out


def _outcome_label(
    *,
    val_bpb: float | None,
    success: bool | None,
    best_val: float | None,
) -> str:
    if success is False or (val_bpb is None and success is not True):
        return "failed"
    if val_bpb is None:
        return "not_run"
    if best_val is None:
        return "ran"
    delta = float(val_bpb) - best_val
    if delta <= 0:
        return "new_best"
    return "regression"


def _vs_best_text(val_bpb: float | None, best_val: float | None) -> str:
    if val_bpb is None or best_val is None:
        return "—"
    delta = float(val_bpb) - best_val
    if abs(delta) < 1e-12:
        return "0 (ties best)"
    sign = "+" if delta > 0 else ""
    quality = "worse" if delta > 0 else "better"
    return f"{sign}{delta:.6g} ({quality})"


async def load_research_history(memory_key: str) -> dict[str, Any]:
    """Merge saved edits, run results, and outcomes for cross-session agent context."""
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    code_index: list[dict[str, Any]] = await memory.read_json.aio("memory/code_index.json", default=[])
    leaderboard: list[dict[str, Any]] = await memory.read_json.aio("memory/leaderboard.json", default=[])
    promising: list[dict[str, Any]] = await memory.read_json.aio("memory/promising_code.json", default=[])
    hypotheses: list[dict[str, Any]] = await memory.read_json.aio("memory/hypotheses.json", default=[])

    lb_by_title = _latest_by_title(leaderboard)
    prom_by_title = _latest_by_title(promising)
    hyp_by_title = _latest_by_title(hypotheses)

    best_val, best_title = _best_from_entries(leaderboard)
    if best_val is None:
        best_val, best_title = _best_from_entries(promising)

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()

    for edit in code_index:
        title = str(edit.get("title", ""))
        key = _title_key(title)
        if not key:
            continue
        seen.add(key)
        lb = lb_by_title.get(key, {})
        prom = prom_by_title.get(key, {})
        hyp = hyp_by_title.get(key, {})

        val = lb.get("val_bpb")
        if val is None:
            val = prom.get("val_bpb")
        val_f = float(val) if val is not None else None

        success = lb.get("success")
        if success is None and val_f is not None:
            success = True
        if success is None and lb.get("error"):
            success = False

        beat_best = val_f is not None and best_val is not None and val_f <= best_val
        trials.append(
            {
                "title": title,
                "change_summary": edit.get("change_summary") or prom.get("change_summary") or "",
                "edited_at": edit.get("edited_at"),
                "val_bpb": val_f,
                "model_name": lb.get("model_name"),
                "success": success,
                "error": lb.get("error"),
                "hypothesis": hyp.get("hypothesis"),
                "expected_effect": hyp.get("expected_effect"),
                "beat_best": beat_best,
                "delta_vs_best": (val_f - best_val) if val_f is not None and best_val is not None else None,
                "vs_best": _vs_best_text(val_f, best_val),
                "outcome": _outcome_label(val_bpb=val_f, success=success, best_val=best_val),
                "kept": bool(prom.get("kept")),
            }
        )

    for key, lb in lb_by_title.items():
        if key in seen:
            continue
        val = lb.get("val_bpb")
        val_f = float(val) if val is not None else None
        success = lb.get("success")
        if success is None and val_f is not None:
            success = True
        trials.append(
            {
                "title": lb.get("title", key),
                "change_summary": "",
                "edited_at": None,
                "val_bpb": val_f,
                "model_name": lb.get("model_name"),
                "success": success,
                "error": lb.get("error"),
                "hypothesis": hyp_by_title.get(key, {}).get("hypothesis"),
                "expected_effect": hyp_by_title.get(key, {}).get("expected_effect"),
                "beat_best": val_f is not None and best_val is not None and val_f <= best_val,
                "delta_vs_best": (val_f - best_val) if val_f is not None and best_val is not None else None,
                "vs_best": _vs_best_text(val_f, best_val),
                "outcome": _outcome_label(val_bpb=val_f, success=success, best_val=best_val),
                "kept": bool(prom_by_title.get(key, {}).get("kept")),
            }
        )

    trials.sort(key=lambda t: (t.get("edited_at") or "", t.get("title", "")))

    return {
        "memory_key": memory_key,
        "best_val_bpb": best_val,
        "best_title": best_title,
        "trials": trials,
        "count_edits": len(code_index),
        "count_runs": sum(1 for t in trials if t.get("val_bpb") is not None or t.get("success") is False),
        "count_regressions": sum(1 for t in trials if t.get("outcome") == "regression"),
        "count_new_best": sum(1 for t in trials if t.get("outcome") == "new_best"),
    }


def format_research_history_for_directive(history: dict[str, Any], *, max_rows: int = 20) -> str:
    """Render prior edits/results as a compact block for the run directive."""
    trials: list[dict[str, Any]] = history.get("trials") or []
    if not trials:
        return ""

    best_val = history.get("best_val_bpb")
    best_title = history.get("best_title")
    header = "\n\n## Prior research (from memory — continue, do not repeat)\n"
    if best_val is not None:
        header += f"Current best: **val_bpb={best_val:.6g}** ({best_title}). Lower is better.\n"
    else:
        header += "No successful runs recorded yet.\n"

    header += (
        "Call ``get_code_edit_history()`` at the start to refresh this table. "
        "Use ``read_train_code`` on the best title to fork winners.\n\n"
    )

    lines = [
        "| Title | Change | val_bpb | vs best | Outcome |",
        "| --- | --- | --- | --- | --- |",
    ]
    for trial in trials[-max_rows:]:
        title = str(trial.get("title", ""))
        change = str(trial.get("change_summary", ""))[:72]
        val = trial.get("val_bpb")
        val_s = f"{float(val):.6g}" if val is not None else ("failed" if trial.get("success") is False else "—")
        lines.append(
            f"| {title} | {change} | {val_s} | {trial.get('vs_best', '—')} | {trial.get('outcome', '—')} |"
        )

    omitted = len(trials) - max_rows
    footer = ""
    if omitted > 0:
        footer = f"\n({omitted} older trial(s) omitted — use get_code_edit_history for the full list.)\n"

    return header + "\n".join(lines) + footer
