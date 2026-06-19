"""Resilient LLM callback for Flyte agents."""

from __future__ import annotations

import asyncio
from typing import Any

import flyte
from flyte.ai.agents import LLMMessage
from flyte.ai.agents._llm import _default_call_llm

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
