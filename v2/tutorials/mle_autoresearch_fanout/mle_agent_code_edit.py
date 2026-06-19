"""Autoresearch with a code-editing MLE agent.

Like :mod:`mle_agent`, but the agent edits ``train.py`` directly (karpathy/autoresearch
style) instead of passing structured hyperparameters. Each experiment:

1. ``edit_train_code`` — save a full ``train.py`` edit to :class:`MemoryStore`
2. ``run_experiment`` — execute it in a ``unionai-sandbox`` session via
   ``async with sb.on_device.session(backend="userns")``
3. Self-healing — a ``call_handler`` right-sizes the Flyte task and retries when
   sandbox ``stderr`` indicates OOM (exit 137, "Killed", "OutOfMemory", …)

Run::

    uv run python mle_agent_code_edit.py --n-experiments 4 --max-turns 50
"""

from __future__ import annotations

import dataclasses
from typing import Any

from autoresearch_types import (
    AutoresearchOutput,
    DEFAULT_NUM_SHARDS,
    MAX_DEVICE_BATCH_SIZE,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
)
from bundle import agent_env, build_bundle, experiment_env, materialize_cache, profile_bundle
from call_handlers_sandbox import heal_sandbox_oom
from llm_call import call_llm
from code_edit_tools import (
    MEMORY_KEY,
    check_duplicate_config,
    edit_train_code,
    get_baseline_train_code,
    get_code_edit_history,
    get_promising_code,
    load_saved_code_edits,
    load_config_overrides,
    load_train_code,
    read_train_code,
    record_experiment_result,
    record_promising_run,
    register_config_signature,
)
from report import (
    render_activity_log,
    render_code_edits_panel,
    render_leaderboard,
    render_memory_panel,
    render_summary,
)
from reports import parse_leaderboard
from reports_code_edit import directive_code_edit
from research_history import load_research_history
from research_tools import (
    compare_experiments,
    get_leaderboard,
    inspect_dataset,
    record_hypothesis,
    search_arxiv,
)
from sandbox_runner import run_train_in_sandbox

import flyte
import flyte.report
from flyte.ai.agents import Agent, MemoryStore, agent_progress_cb, tool

MODEL = "claude-sonnet-4-5"


@tool(call_handler=heal_sandbox_oom)
@experiment_env.task
async def run_experiment(
    title: str,
    time_budget_sec: int = 45,
    memory_key: str = MEMORY_KEY,
) -> dict:
    """Train using the agent-edited ``train.py`` for this title inside a sandbox.

    Loads the saved edit from memory (see ``edit_train_code``), materializes the
    climbmix bundle, and runs the code under ``unionai-sandbox`` with
    ``sb.on_device.session(backend="userns")``. On success returns ``val_bpb`` and
    related metrics; on failure returns ``stderr`` so you can fix the code. OOM
    is detected from sandbox stderr and healed by the platform (more memory, retry).

    Args:
        title: Experiment title whose edited ``train.py`` to run.
        time_budget_sec: Wall-clock training budget passed to ``run_training``.
        memory_key: Memory namespace from your directive.

    Returns:
        A dict with ``success``, ``val_bpb`` (if successful), ``stderr`` on failure,
        and ``oom`` when the sandbox was likely OOM-killed.
    """
    train_py = await load_train_code(memory_key, title)
    config_overrides = await load_config_overrides(memory_key, title)
    duplicate = await check_duplicate_config(memory_key, title, train_py, config_overrides)
    if duplicate:
        result = {
            "success": False,
            "title": title,
            "error": (
                f"Duplicate config of '{duplicate['duplicate_of']}' "
                f"(signature {duplicate['config_signature']}); change train.py or overrides."
            ),
            "duplicate_of": duplicate["duplicate_of"],
        }
        await record_experiment_result(memory_key, result)
        return result
    bundle = await build_bundle()
    cache_dir = await materialize_cache(bundle)

    result = await run_train_in_sandbox(
        cache_dir,
        train_py,
        title=title,
        time_budget_sec=time_budget_sec,
        config_overrides=config_overrides or None,
    )

    if result.get("success"):
        await record_promising_run(memory_key, title, result)
        await register_config_signature(memory_key, title, train_py, config_overrides)

    await record_experiment_result(memory_key, result)
    return result


INSTRUCTIONS = f"""\
You are a senior ML-engineer agent running karpathy/autoresearch-style autonomous
research by **editing train.py**, not by passing hyperparameter tool args. Your
goal: find a TinyGPT training script that MINIMIZES val_bpb (LOWER is better).

Available tools:
- get_baseline_train_code — read the starting train.py once.
- get_code_edit_history — **call first on resumed sessions**: all prior edits, val_bpb, vs-best deltas.
- edit_train_code — save a full edited train.py for an experiment title.
- read_train_code / get_promising_code — recall prior edits and what worked.
- inspect_dataset, search_arxiv — same as the structured agent.
- record_hypothesis, get_leaderboard, compare_experiments — bookkeeping.
- run_experiment — execute your edited train.py in a sandbox (expensive step).

How to work:
0. If your directive includes prior research, call get_code_edit_history() immediately and
   read_train_code on the current best title before proposing new experiments.
1. Call get_baseline_train_code and inspect_dataset; optionally search_arxiv.
2. For each experiment: edit_train_code (keep ``run_training(config)``), record_hypothesis,
   then run_experiment with the same title.
3. If run_experiment fails, read ``stderr`` in the result and edit_train_code to fix it.
4. Iterate on promising edits from get_promising_code; discard regressions.
5. After N experiments, STOP with a summary: best val_bpb, what code changes helped, and
   which promising edit to continue from.

Experiment diversity (required):
- Make **each experiment a different hypothesis** — do not repeat the same edit under a new title.
- Vary **one or two knobs at a time** in ``ExperimentConfig`` inside ``run_training``:
  ``n_layer`` (depth), ``n_head``, ``n_embd`` (width), ``dropout``, ``device_batch_size``,
  ``learning_rate``.
- Spread ideas across runs, e.g. deeper vs wider vs higher LR vs lower LR vs more dropout vs
  larger batch — pick axes that are **orthogonal**, not tiny tweaks of the same idea.
- In ``change_summary`` and ``record_hypothesis``, state clearly which knob(s) you changed and why.
- After a promising result, fork from that code but still explore **new directions** (do not
  run five copies of the same architecture with only the title changed).

Keep ``run_training(config: ExperimentConfig) -> ExperimentResult`` in every edit.
Workshop limits: n_layer<={MAX_N_LAYER}, n_embd<={MAX_N_EMBD}, n_head<={MAX_N_HEAD},
device_batch_size<={MAX_DEVICE_BATCH_SIZE}, seq_len=512. Baseline is 3×128, batch 2.
You do NOT size compute — the platform right-sizes each sandbox run and retries on OOM.
"""

DEFAULT_MAX_TURNS = 500


def build_agent(*, max_turns: int = DEFAULT_MAX_TURNS) -> Agent:
    """Construct the code-edit agent with a configurable turn budget."""
    return Agent(
        name="mle-autoresearch-code-agent",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[
            search_arxiv,
            inspect_dataset,
            get_baseline_train_code,
            get_code_edit_history,
            edit_train_code,
            read_train_code,
            get_promising_code,
            record_hypothesis,
            get_leaderboard,
            compare_experiments,
            run_experiment,
        ],
        max_turns=max_turns,
        call_llm=call_llm,
    )


agent = build_agent()


@agent_env.task(report=True)
async def mle_autoresearch_code_agent(
    n_experiments: int = 4,
    num_shards: int = DEFAULT_NUM_SHARDS,
    memory_key: str = MEMORY_KEY,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AutoresearchOutput:
    """Drive the code-edit MLE agent with sandbox execution and live reports."""
    bundle = await build_bundle(num_shards=num_shards)
    profile = await profile_bundle(bundle)

    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    persisted = await memory.read_json.aio("memory/leaderboard.json", default=[])
    promising = await memory.read_json.aio("memory/promising_code.json", default=[])
    history = await load_research_history(memory_key)
    flyte.logger.info(
        "Restored %d messages, %d experiments, %d promising edits, best val_bpb=%s.",
        len(memory.messages),
        len(persisted),
        len(promising),
        history.get("best_val_bpb"),
    )

    events: list[dict[str, Any]] = []

    async def on_event(ev) -> None:
        events.append({"type": ev.type, "data": ev.data})
        if ev.type in ("tool_start", "tool_end", "tool_error", "turn_start", "agent_end"):
            tab = flyte.report.get_tab("Activity")
            tab.replace(render_activity_log(events))
            await flyte.report.flush.aio()
        if ev.type == "tool_end" and ev.data.get("tool") in (
            "edit_train_code",
            "edit_train_code_batch",
            "<sandbox>",
        ):
            edits = await load_saved_code_edits(memory_key)
            if edits:
                flyte.report.get_tab("Code edits").replace(render_code_edits_panel(edits))
                await flyte.report.flush.aio()

    directive_text = directive_code_edit(
        n_experiments,
        profile,
        memory_key,
        history=history,
    )

    token = agent_progress_cb.set(on_event)
    run_agent = build_agent(max_turns=max_turns)
    try:
        result = await run_agent.run.aio(directive_text, memory=memory)
    finally:
        agent_progress_cb.reset(token)

    leaderboard, best = parse_leaderboard(memory.messages)
    leaderboard_dicts = [dataclasses.asdict(e) for e in leaderboard]
    code_edits = await load_saved_code_edits(memory_key)

    tab_lb = flyte.report.get_tab("Leaderboard")
    tab_lb.replace(render_leaderboard(leaderboard, best))

    flyte.report.get_tab("Code edits").replace(
        render_code_edits_panel(code_edits, best_title=best.title if best else None)
    )

    await memory.write_json.aio(
        "memory/leaderboard.json",
        leaderboard_dicts,
        actor="mle-autoresearch-code-agent",
        reason=f"leaderboard after {len(leaderboard)} experiments",
    )
    await memory.save.aio()
    audit = await memory.audit_tail(20)
    hypotheses = await memory.read_json.aio("memory/hypotheses.json", default=[])
    promising = await memory.read_json.aio("memory/promising_code.json", default=[])

    tab_mem = flyte.report.get_tab("Memory")
    tab_mem.replace(
        render_memory_panel(
            memory_key,
            len(memory.messages),
            leaderboard_dicts,
            audit,
            hypotheses,
            persisted_promising=promising,
            code_edits=code_edits,
        )
    )

    await flyte.report.replace.aio(
        render_summary(
            directive_text,
            leaderboard,
            best,
            result.summary or result.error or "",
            code_edits=code_edits,
        )
    )
    await flyte.report.flush.aio()

    return AutoresearchOutput(
        directive=directive_text,
        dataset_profile=profile,
        best=best,
        leaderboard=leaderboard,
        summary=result.summary or result.error or "",
        memory_key=memory_key,
        total_experiments=len(leaderboard),
    )


if __name__ == "__main__":
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experiments", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--memory-key", default=MEMORY_KEY)
    parser.add_argument(
        "--config",
        default=os.environ.get("FLYTE_CONFIG", os.path.expanduser("~/.flyte/config.yaml")),
    )
    args = parser.parse_args()

    flyte.init_from_config(args.config, image_builder="remote")

    async def main() -> None:
        run = await flyte.with_runcontext(copy_style="all").run.aio(
            mle_autoresearch_code_agent,
            n_experiments=args.n_experiments,
            num_shards=args.num_shards,
            memory_key=args.memory_key,
            max_turns=args.max_turns,
        )
        print(f"View run at: {run.url}")

    asyncio.run(main())
