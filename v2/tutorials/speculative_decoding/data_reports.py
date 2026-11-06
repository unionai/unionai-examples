"""Reports for data synthesis: sampled prompts, generated shards, and the final mix."""

import statistics
from collections import Counter
from html import escape

import report_ui as ui


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


def prompts_report(chat: list[dict], agent: list[dict], num_shards: int) -> str:
    chat_lens = [len(c["messages"][0]["content"]) for c in chat]
    tools = [len(a["tools"]) for a in agent]
    return ui.page(
        ui.hero(
            "Sampled prompts",
            "Inputs for target-model generation, split into shards that run in parallel.",
            ["UltraChat 200k", "ToolACE", f"{num_shards} shards"],
        ),
        ui.tiles({
            "chat prompts": _fmt(len(chat)),
            "agent tasks": _fmt(len(agent)),
            "tools per agent task (mean)": f"{statistics.mean(tools):.1f}" if tools else "0",
            "recorded tool results": _fmt(sum(len(a["tool_results"]) for a in agent)),
        }),
        ui.charts(
            ui.histogram("Chat prompt length (characters)", chat_lens),
            ui.histogram("Tools offered per agent task", tools, bins=max(min(len(set(tools)), 12), 1)),
        ),
        ui.section(
            "Example agent task",
            ui.chat([{"role": "user", "content": agent[0]["request"]}]) if agent else "",
            ui.table(
                ["tool", "description"],
                [[t["function"]["name"], t["function"]["description"][:120]] for t in agent[0]["tools"]],
                numeric=set(),
            ) if agent else "",
        ),
    )


def chat_shard_report(shard, rows: list[dict]) -> str:
    return ui.page(
        ui.hero(
            "Regenerated chat shard",
            "UltraChat prompts answered by Qwen3-8B. Only target-written tokens are trained on.",
            ["regenerate-responses", "temperature 0.7"],
        ),
        ui.tiles(
            {
                "training rows": _fmt(shard.rows),
                "target tokens": _fmt(shard.target_tokens),
                "total tokens": _fmt(shard.tokens),
                "failed prompts": _fmt(shard.failed),
            },
            tone={"failed prompts": "bad"} if shard.failed else {},
        ),
        ui.charts(ui.histogram("Target tokens per row", shard.lengths)),
        ui.section(
            "Samples",
            *(f"<div class='chat'><div class='msg assistant'>{escape(r['text'][-1500:])}</div></div>" for r in rows[:2]),
            lede="The tail of each training row, as the trainer sees it after the chat template.",
        ),
    )


def trace_stats(trace: dict, tokenizer) -> dict:
    """Token and tool-call counts for one agent episode."""
    messages = trace["messages"]
    ids = tokenizer.apply_chat_template(
        messages, tools=trace["tools"], tokenize=True, enable_thinking=False
    )
    target = 0
    calls = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        text = m.get("content") or ""
        for c in m.get("tool_calls", []):
            calls.append(c["function"]["name"])
            text += c["function"]["name"] + c["function"].get("arguments", "")
        target += len(tokenizer.encode(text))
    return {
        "tokens": len(ids["input_ids"] if isinstance(ids, dict) else ids),
        "target_tokens": target,
        "tool_calls": len(calls),
        "tools_called": calls,
        "turns": sum(m["role"] == "assistant" for m in messages),
    }


def agent_shard_report(shard, traces: list[dict], stats: list[dict]) -> str:
    calls = Counter(name for s in stats for name in s["tools_called"])
    with_calls = sum(s["tool_calls"] > 0 for s in stats)
    return ui.page(
        ui.hero(
            "Agent trace shard",
            "ToolACE tasks run end to end against Qwen3-8B, with recorded tool results replayed.",
            ["tool calling", "Hermes parser", "multi-turn"],
        ),
        ui.tiles(
            {
                "episodes": _fmt(shard.rows),
                "episodes with tool calls": f"{with_calls / max(len(stats), 1):.0%}",
                "tool calls": _fmt(shard.tool_calls),
                "target tokens": _fmt(shard.target_tokens),
                "failed episodes": _fmt(shard.failed),
            },
            tone={"failed episodes": "bad"} if shard.failed else {},
        ),
        ui.charts(
            ui.histogram("Assistant turns per episode", [s["turns"] for s in stats], bins=6),
            ui.bar_chart("Most called tools", dict(calls.most_common(8)), fmt="{:.0f}"),
        ),
        ui.section("Example episode", ui.chat(traces[0]["messages"]) if traces else ""),
    )


def dataset_report(data) -> str:
    chat_tokens = sum(s.target_tokens for s in data.chat)
    agent_tokens = sum(s.target_tokens for s in data.agent)
    rows = [
        [f"{s.kind} {i}", _fmt(s.rows), _fmt(s.target_tokens), _fmt(s.tool_calls), _fmt(s.failed)]
        for i, s in enumerate(data.chat + data.agent)
    ]
    return ui.page(
        ui.hero(
            "Draft model training set",
            "Every token below was written by Qwen3-8B, so the draft head learns the target's own distribution.",
            ["target-generated", f"{len(data.chat) + len(data.agent)} shards"],
        ),
        ui.tiles({
            "chat rows": _fmt(sum(s.rows for s in data.chat)),
            "agent episodes": _fmt(sum(s.rows for s in data.agent)),
            "target tokens": _fmt(chat_tokens + agent_tokens),
            "tool calls": _fmt(sum(s.tool_calls for s in data.agent)),
        }),
        ui.charts(
            ui.bar_chart("Target tokens by source", {"chat": chat_tokens, "agent": agent_tokens}, fmt="{:,.0f}"),
            ui.histogram(
                "Target tokens per example",
                [n for s in data.chat + data.agent for n in s.lengths],
            ),
        ),
        ui.section("Shards", ui.table(["shard", "rows", "target tokens", "tool calls", "failed"], rows)),
    )



def traffic_report(traces: list[dict], failed: int, latencies: list[float], title: str = "Agent traffic") -> str:
    """Traces an application produced: what the agent did and what it cost in time."""
    calls = Counter(
        c["function"]["name"]
        for t in traces
        for m in t["messages"]
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
    )
    turns = [sum(m["role"] == "assistant" for m in t["messages"]) for t in traces]
    return ui.page(
        ui.hero(
            title,
            "Each episode is a trace: the conversation, tool schemas, and every model response. "
            "These traces are the next training set for the draft head.",
            ["flyte.trace", "OpenAI chat format"],
        ),
        ui.tiles(
            {
                "episodes": _fmt(len(traces)),
                "failed": _fmt(failed),
                "tool calls": _fmt(sum(calls.values())),
                "episode latency p50": f"{statistics.median(latencies):.1f}s" if latencies else "-",
            },
            tone={"failed": "bad"} if failed else {},
        ),
        ui.charts(
            ui.histogram("Episode latency (s)", latencies, unit="s"),
            ui.histogram("Assistant turns per episode", turns, bins=6),
            ui.bar_chart("Most called tools", dict(calls.most_common(8)), fmt="{:.0f}"),
        ),
        ui.section("Example episode", ui.chat(traces[0]["messages"]) if traces else ""),
    )
