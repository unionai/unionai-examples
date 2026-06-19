"""Self-healing ``call_handler`` for sandbox-backed training runs.

OOM is healed two ways (both re-prompt the LLM with failure context):

1. **Flyte pod OOM** — :class:`flyte.errors.OOMError` from ``tool_fn.target.override(...).aio(...)``
2. **Sandbox subprocess OOM** — ``result["oom"]`` from sandbox ``stderr`` / exit 137
"""

from __future__ import annotations

import json
import re
from typing import Any

import flyte
import flyte.errors
from flyte.ai.agents import LLMCallable, ToolFn

from call_handlers import (
    RESOURCE_CEILING,
    _ensure_oom_increase,
    _extract_json,
    _resources_from_spec,
    _sizing_payload,
)
from autoresearch_types import MAX_DEVICE_BATCH_SIZE, MAX_N_EMBD, MAX_N_LAYER
import prepare
from code_edit_tools import load_train_code

CODE_SIZING_SYSTEM = """\
You are a Kubernetes capacity planner for a PyTorch language-model training task.
The agent edits ``train.py`` directly (karpathy/autoresearch style). Given the
experiment title, training time budget, and a summary of the edited code, estimate
the *minimum sensible* compute so the sandbox run finishes without being OOM-killed.

Key facts:
- Sequence length is fixed at {seq_len}; attention memory scales with batch × heads.
- Deeper/wider models and larger ``device_batch_size`` in the code need more RAM.
- This cluster is CPU-only; do not request GPUs.
- Workshop caps: n_layer<={max_layer}, n_embd<={max_embd}, device_batch_size<={max_batch}.

Cluster limits (never exceed): cpu <= {{max_cpu}}, memory <= {{max_memory}}.

Respond with ONLY a JSON object (no prose, no code fences), any of these keys:
  - "cpu":    cores, e.g. 2, 4, 8
  - "memory": a Kubernetes memory string, e.g. "2Gi", "4Gi", "8Gi", "16Gi"
Example: {{"cpu": 4, "memory": "8Gi"}}
"""


def _code_sizing_system_prompt() -> str:
    max_cpu = int(RESOURCE_CEILING.cpu or 16)
    max_memory = RESOURCE_CEILING.memory if isinstance(RESOURCE_CEILING.memory, str) else "32Gi"
    return CODE_SIZING_SYSTEM.format(
        seq_len=prepare.MAX_SEQ_LEN,
        max_layer=MAX_N_LAYER,
        max_embd=MAX_N_EMBD,
        max_batch=MAX_DEVICE_BATCH_SIZE,
        max_cpu=max_cpu,
        max_memory=max_memory,
    )


def _code_hints(train_py: str) -> dict[str, Any]:
    hints: dict[str, Any] = {"code_lines": len(train_py.splitlines())}
    for name, pattern in (
        ("n_layer", r"n_layer\s*=\s*(\d+)"),
        ("n_embd", r"n_embd\s*=\s*(\d+)"),
        ("device_batch_size", r"device_batch_size\s*=\s*(\d+)"),
    ):
        match = re.search(pattern, train_py)
        if match:
            hints[name] = int(match.group(1))
    return hints


async def right_size_code(
    call_llm: LLMCallable,
    model: str,
    kwargs: dict[str, Any],
    *,
    previous_resources: flyte.Resources | None = None,
    oom_error: str | None = None,
    attempt: int = 0,
) -> flyte.Resources:
    memory_key = kwargs.get("memory_key", "mle-autoresearch-code")
    title = str(kwargs.get("title", "experiment"))
    train_py = await load_train_code(memory_key, title)
    experiment = {
        "title": title,
        "time_budget_sec": kwargs.get("time_budget_sec", 45),
        **_code_hints(train_py),
    }
    user = json.dumps(
        _sizing_payload(
            experiment,
            previous_resources=previous_resources,
            oom_error=oom_error,
            attempt=attempt,
        ),
        default=str,
    )
    try:
        reply = await call_llm(
            model,
            _code_sizing_system_prompt(),
            [{"role": "user", "content": user}],
            None,
        )
        spec = _extract_json(reply.content)
    except Exception as exc:  # pragma: no cover
        flyte.logger.warning("Code right-sizing failed (%s); using floor.", exc)
        spec = {}
    resources = _resources_from_spec(spec)
    if previous_resources is not None and oom_error:
        resources = _ensure_oom_increase(resources, previous_resources)
    flyte.logger.info("right-size %s -> %s", title, resources)
    return resources


def sandbox_oom_recovery_handler(*, max_oom_retries: int = 3):
    """Right-size the Flyte task, run the sandbox tool, heal OOM via stderr inspection."""

    async def handle(call_llm: LLMCallable, tool_fn: ToolFn, **kwargs: Any) -> Any:
        resources = await right_size_code(call_llm, tool_fn.model, kwargs)
        attempt = 0
        while True:
            try:
                with flyte.group(f"{kwargs.get('title', 'experiment')}-attempt-{attempt + 1}"):
                    sized = tool_fn.target.override(resources=resources)
                    result = await sized.aio(**kwargs)
            except flyte.errors.OOMError as exc:
                if attempt >= max_oom_retries:
                    flyte.logger.error(
                        "run_experiment Flyte OOM after %d retries; giving up.",
                        attempt,
                    )
                    raise
                attempt += 1
                resources = await right_size_code(
                    call_llm,
                    tool_fn.model,
                    kwargs,
                    previous_resources=resources,
                    oom_error=str(exc),
                    attempt=attempt,
                )
                flyte.logger.warning(
                    "run_experiment Flyte OOM; retrying with cpu=%s memory=%s",
                    resources.cpu,
                    resources.memory,
                )
                continue

            if isinstance(result, dict):
                result["resources"] = f"cpu={resources.cpu}, mem={resources.memory}"
                result["oom_retries"] = attempt

            if isinstance(result, dict) and result.get("oom"):
                if attempt >= max_oom_retries:
                    flyte.logger.error(
                        "run_experiment sandbox OOM after %d retries; stderr=%s",
                        attempt,
                        (result.get("stderr") or "")[:500],
                    )
                    return result
                attempt += 1
                stderr = str(result.get("stderr") or "sandbox OOM detected")
                resources = await right_size_code(
                    call_llm,
                    tool_fn.model,
                    kwargs,
                    previous_resources=resources,
                    oom_error=stderr,
                    attempt=attempt,
                )
                flyte.logger.warning(
                    "run_experiment sandbox OOM (stderr); retrying with cpu=%s memory=%s",
                    resources.cpu,
                    resources.memory,
                )
                continue

            return result

    return handle


heal_sandbox_oom = sandbox_oom_recovery_handler(max_oom_retries=3)
