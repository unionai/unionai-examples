"""Prompt sources and the tool-calling agent used throughout the tutorial.

Three kinds of traffic matter for a draft model:

- General chat (UltraChat, MT-Bench): what public draft heads are trained on.
- Math (GSM8K): a task with a checkable answer, used to show accuracy is preserved.
- Agent traffic (ToolACE): multi-turn tool calling with JSON arguments and tool results.
  This is the workload the tutorial adapts the draft model to.

ToolACE ships recorded tool results, so the agent can run against the real target model
while tool execution is replayed from the dataset.
"""

import json
import random
import re
from typing import Any, Awaitable, Callable

import httpx

from config import TARGET_MODEL

AGENT_SYSTEM_PROMPT = (
    "You are an assistant that completes the user's request by calling the provided tools. "
    "Call a tool whenever it can answer part of the request, then summarize the results "
    "for the user."
)

# Qwen3 is a reasoning model. Agent and chat traffic in this tutorial runs without the
# thinking phase, the common setting for latency-sensitive applications.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


# {{docs-fragment toolace}}
def _tool_name(name: str) -> str:
    """OpenAI tool names allow only letters, digits, `_`, and `-`."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]


def _to_openai_schema(schema: Any) -> Any:
    """ToolACE uses `"type": "dict"`; JSON Schema calls it `object`."""
    if isinstance(schema, dict):
        out = {k: _to_openai_schema(v) for k, v in schema.items() if v is not None}
        if out.get("type") == "dict":
            out["type"] = "object"
        return out
    if isinstance(schema, list):
        return [_to_openai_schema(v) for v in schema]
    return schema


def parse_toolace(row: dict) -> dict | None:
    """Turn a ToolACE row into an agent task: tools, a user request, and replayable results."""
    match = re.search(r"\[\s*\{.*\}\s*\]", row["system"], flags=re.DOTALL)
    if not match:
        return None
    try:
        functions = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    tools = [
        {
            "type": "function",
            "function": {
                "name": _tool_name(fn["name"]),
                "description": fn.get("description", ""),
                "parameters": _to_openai_schema(fn.get("parameters") or {"type": "object"}),
            },
        }
        for fn in functions
        if isinstance(fn, dict) and "name" in fn
    ]
    turns = row["conversations"]
    if not tools or not turns or turns[0]["from"] != "user":
        return None
    tool_results = [t["value"] for t in turns if t["from"] == "tool"]
    return {"tools": tools, "request": turns[0]["value"], "tool_results": tool_results}
# {{/docs-fragment toolace}}


# ToolACE is split three ways so no evaluation prompt is ever trained on:
# initial training data, simulated production traffic, and held-out evaluation.
AGENT_SPLITS = {"train": (0.0, 0.6), "traffic": (0.6, 0.8), "heldout": (0.8, 1.0)}


def load_agent_tasks(split: str, n: int, seed: int = 0) -> list[dict]:
    """Agent tasks from one ToolACE split: "train", "traffic", or "heldout"."""
    from datasets import load_dataset

    rows = load_dataset("Team-ACE/ToolACE", split="train")
    lo, hi = AGENT_SPLITS[split]
    indices = list(range(int(len(rows) * lo), int(len(rows) * hi)))
    random.Random(seed).shuffle(indices)
    tasks = []
    for i in indices:
        task = parse_toolace(rows[i])
        if task:
            tasks.append(task)
        if len(tasks) == n:
            break
    return tasks


def load_chat_prompts(n: int, seed: int = 0) -> list[list[dict]]:
    """First-turn UltraChat conversations, as chat messages."""
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    indices = random.Random(seed).sample(range(len(rows)), n)
    return [[{"role": "user", "content": rows[i]["prompt"]}] for i in indices]


def load_mt_bench() -> list[list[dict]]:
    """The 80 first-turn MT-Bench prompts."""
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    return [[{"role": "user", "content": r["prompt"][0]}] for r in rows]


def load_gsm8k(n: int) -> list[dict]:
    """GSM8K test questions with their reference answers."""
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": r["question"] + "\nGive the final answer after '####'.",
                }
            ],
            "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for r in rows
    ]


def gsm8k_correct(output: str, answer: str) -> bool:
    """Compare the last number in the output with the reference answer."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", output.split("####")[-1])
    return bool(numbers) and numbers[-1].replace(",", "").rstrip(".") == answer


# {{docs-fragment agent}}
Complete = Callable[[list[dict], list[dict]], Awaitable[dict]]


def http_complete(
    client: httpx.AsyncClient,
    base_url: str,
    *,
    api_key: str | None = None,
    model: str = TARGET_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Complete:
    """An OpenAI-compatible chat completion call that returns the assistant message."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def complete(messages: list[dict], tools: list[dict]) -> dict:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **NO_THINKING,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    return complete


async def run_agent(complete: Complete, task: dict, max_rounds: int = 3) -> dict:
    """Run one tool-calling episode and return it as a trace.

    The trace is the full conversation in OpenAI chat format plus the tool schemas, which
    is exactly what a draft model trains on.
    """
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task["request"]},
    ]
    replay = list(task["tool_results"])
    for _ in range(max_rounds):
        message = await complete(messages, task["tools"])
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            assistant["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": c["function"]}
                for c in message["tool_calls"]
            ]
        messages.append(assistant)
        if not message.get("tool_calls"):
            break
        # Replay the recorded tool output instead of calling a live API.
        for call in message["tool_calls"]:
            result = replay.pop(0) if replay else json.dumps({"status": "ok"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    return {"messages": messages, "tools": task["tools"]}
# {{/docs-fragment agent}}
