"""Tool ``call_handler`` helpers for the autoresearch MLE agent.

A ``call_handler`` wraps every invocation of an agent tool. These helpers
right-size Flyte task resources from the tool arguments and self-heal from
:class:`flyte.errors.OOMError` by re-prompting the LLM with the failure context.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

import flyte
import flyte.errors
from flyte.ai.agents import LLMCallable, ToolFn

from autoresearch_types import MAX_DEVICE_BATCH_SIZE, MAX_N_EMBD, MAX_N_LAYER
import prepare

RESOURCE_SIZING_SYSTEM = """\
You are a Kubernetes capacity planner for a PyTorch language-model training task.
Given the experiment config (TinyGPT depth/width/heads, batch size), estimate the
*minimum sensible* compute so the run finishes without being OOM-killed, while not
wildly over-provisioning.

Key facts about this task:
- Sequence length is FIXED at {seq_len}. The attention matrix dominates memory and
  scales with device_batch_size * n_head * {seq_len}^2.
- Parameters + activations scale with n_layer and n_embd^2.
- This cluster is CPU-only for the workshop; do not request GPUs.
- Workshop caps: n_layer<={max_layer}, n_embd<={max_embd}, device_batch_size<={max_batch}.

Cluster limits (never exceed): cpu <= {{max_cpu}}, memory <= {{max_memory}}.

Respond with ONLY a JSON object (no prose, no code fences), any of these keys:
  - "cpu":    cores, e.g. 2, 4, 8
  - "memory": a Kubernetes memory string, e.g. "2Gi", "4Gi", "8Gi", "16Gi"
Example: {{"cpu": 4, "memory": "8Gi"}}
"""

RESOURCE_FLOOR = flyte.Resources(cpu=2, memory="2Gi")
RESOURCE_CEILING = flyte.Resources(cpu=16, memory="32Gi")
_MEM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$")
_ALLOWED_KEYS = ("cpu", "memory")


def _sizing_system_prompt() -> str:
    max_cpu = int(RESOURCE_CEILING.cpu or 16)
    max_memory = RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
    return RESOURCE_SIZING_SYSTEM.format(
        seq_len=prepare.MAX_SEQ_LEN,
        max_layer=MAX_N_LAYER,
        max_embd=MAX_N_EMBD,
        max_batch=MAX_DEVICE_BATCH_SIZE,
        max_cpu=max_cpu,
        max_memory=max_memory,
    )


def _extract_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _memory_to_mib(memory: str | None) -> int:
    if not memory:
        return 2048
    match = _MEM_RE.match(memory)
    if not match:
        return 2048
    value = float(match.group(1))
    unit = (match.group(2) or "Mi").lower()
    # LLMs often emit GB/MB; Kubernetes expects Gi/Mi. Treat GB as Gi for sizing.
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


def _resources_from_spec(spec: dict[str, Any]) -> flyte.Resources:
    kwargs: dict[str, Any] = {"cpu": RESOURCE_FLOOR.cpu, "memory": RESOURCE_FLOOR.memory}
    for key in _ALLOWED_KEYS:
        value = spec.get(key)
        if value in (None, "", "null"):
            continue
        kwargs[key] = value
    try:
        return _cap_resources(flyte.Resources(**kwargs))
    except Exception as exc:  # pragma: no cover - defensive against bad model output
        flyte.logger.warning("Invalid resource spec %s (%s); using floor.", spec, exc)
        return RESOURCE_FLOOR


def _ensure_oom_increase(resources: flyte.Resources, previous: flyte.Resources) -> flyte.Resources:
    """If the model under-provisioned after OOM, bump deterministically up to the ceiling."""
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
    """Deterministic memory bump after OOM (inline retry loops without an LLM)."""
    return _ensure_oom_increase(resources, resources)


def _sizing_payload(
    args: dict[str, Any],
    *,
    previous_resources: flyte.Resources | None = None,
    oom_error: str | None = None,
    attempt: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"experiment": args}
    if previous_resources is not None:
        payload["previous_resources"] = {
            "cpu": previous_resources.cpu,
            "memory": previous_resources.memory,
        }
    if oom_error:
        payload["oom_error"] = oom_error[:4000]
        payload["retry_attempt"] = attempt
        payload["instruction"] = (
            "The previous allocation OOM-killed this run. Propose a larger allocation "
            "that should fit; stay within cluster limits."
        )
    return payload


@flyte.trace
async def right_size(
    call_llm: LLMCallable,
    model: str,
    args: dict[str, Any],
    *,
    previous_resources: flyte.Resources | None = None,
    oom_error: str | None = None,
    attempt: int = 0,
) -> flyte.Resources:
    """Ask the LLM to size compute for this experiment, optionally after an OOM."""
    user = json.dumps(
        _sizing_payload(
            args,
            previous_resources=previous_resources,
            oom_error=oom_error,
            attempt=attempt,
        ),
        default=str,
    )
    try:
        reply = await call_llm(model, _sizing_system_prompt(), [{"role": "user", "content": user}], None)
        spec = _extract_json(reply.content)
    except Exception as exc:  # pragma: no cover - sizing must never break the tool
        flyte.logger.warning("Right-sizing LLM call failed (%s); using floor.", exc)
        spec = {}
    resources = _resources_from_spec(spec)
    if previous_resources is not None and oom_error:
        resources = _ensure_oom_increase(resources, previous_resources)
    flyte.logger.info("right-size %s -> %s", args.get("title"), resources)
    return resources


def oom_recovery_handler(*, max_oom_retries: int = 3):
    """Build a ``@tool`` ``call_handler`` that right-sizes and heals OOM.

    On every invocation it (1) uses ``call_llm`` to estimate the
    ``flyte.Resources`` for the proposed config, (2) applies them via
    ``tool_fn.target.override(resources=...)``, and (3) on
    :class:`flyte.errors.OOMError`, re-prompts ``call_llm`` with the failure
    context (capped to cluster limits), up to ``max_oom_retries``.
    The number of OOM retries and the final resources are folded into the result
    so the report can show the self-healing in action.
    """

    async def handle(call_llm: LLMCallable, tool_fn: ToolFn, **kwargs: Any) -> Any:
        resources = await right_size(call_llm, tool_fn.model, kwargs)
        attempt = 0
        while True:
            try:
                with flyte.group(f"{kwargs.get('title', 'experiment')}-attempt-{attempt + 1}"):
                    sized = tool_fn.target.override(resources=resources)
                    result = await sized.aio(**kwargs)
            except flyte.errors.OOMError as exc:
                if attempt >= max_oom_retries:
                    flyte.logger.error("run_experiment OOMed after %d retries; giving up.", attempt)
                    raise
                attempt += 1
                resources = await right_size(
                    call_llm,
                    tool_fn.model,
                    kwargs,
                    previous_resources=resources,
                    oom_error=str(exc),
                    attempt=attempt,
                )
                flyte.logger.warning(
                    "run_experiment OOMed; retrying with cpu=%s memory=%s",
                    resources.cpu,
                    resources.memory,
                )
                continue
            if isinstance(result, dict):
                result["resources"] = f"cpu={resources.cpu}, mem={resources.memory}"
                result["oom_retries"] = attempt
            return result

    return handle


heal_oom = oom_recovery_handler(max_oom_retries=3)
