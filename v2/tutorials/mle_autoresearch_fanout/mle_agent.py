"""Autoresearch with an MLE agent — the workshop's central example.

An ML-engineer agent runs `karpathy/autoresearch
<https://github.com/karpathy/autoresearch>`_-style experiments on Flyte: it
proposes a TinyGPT config, trains it for a short budget on the climbmix corpus,
reads back **val_bpb** (lower is better), keeps or discards, and iterates.

It is built entirely from the ``flyte.ai.agents`` toolkit and showcases the four
capabilities that make agents *durable* and *self-healing* on Flyte:

1. **The Agent construct** — :class:`flyte.ai.agents.Agent` drives a plain
   tool-use loop. Its tools include literature search, dataset inspection,
   config validation, and :func:`run_experiment`, an ``@env.task`` that trains
   on-cluster.
2. **Self-healing via a tool ``call_handler``** — every ``run_experiment`` call
   is intercepted by :func:`call_handlers.oom_recovery_handler`, which right-sizes compute for
   the proposed config and, on :class:`flyte.errors.OOMError`, **doubles memory
   and retries**. The agent never thinks about infrastructure.
3. **Memory** — a keyed :class:`flyte.ai.agents.MemoryStore` carries the
   transcript *and* a persisted leaderboard across runs, so the agent resumes
   research where it left off.
4. **Reports** — the agent's progress, the live ``val_bpb`` leaderboard, and the
   contents of its memory are streamed to a Flyte report.

Run::

    export FLYTE_INTERNAL_EXECUTION_PROJECT=flytesnacks  # or use flyte config
    uv run python mle_agent.py --max-turns 50
"""

from __future__ import annotations

import dataclasses
from typing import Any

# NOTE: ``prepare`` and ``train`` pull in torch / rustbpe / pyarrow, so they are
# imported lazily inside the tasks that need them. This keeps local registration
# light (the driver only needs ``flyte`` + ``litellm``) while the heavy stack is
# installed in the task image.
from autoresearch_types import (
    AutoresearchOutput,
    DatasetProfile,
    DEFAULT_NUM_SHARDS,
    ExperimentConfig,
    MAX_DEVICE_BATCH_SIZE,
    MAX_N_EMBD,
    MAX_N_HEAD,
    MAX_N_LAYER,
)
from bundle import (
    agent_env,
    build_bundle,
    experiment_env,
    materialize_cache,
    profile_bundle,
)
from call_handlers import heal_oom
from llm_call import call_llm
from report import (
    render_activity_log,
    render_leaderboard,
    render_memory_panel,
    render_summary,
)
from reports import directive, parse_leaderboard
from research_tools import (
    MEMORY_KEY,
    compare_experiments,
    get_leaderboard,
    inspect_dataset,
    record_hypothesis,
    search_arxiv,
    validate_experiment_config,
)

import flyte
import flyte.report
from flyte.ai.agents import Agent, MemoryStore, agent_progress_cb, tool

MODEL = "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Self-healing tool: train one experiment, right-sizing + healing OOM per call
# ---------------------------------------------------------------------------


@tool(call_handler=heal_oom)
@experiment_env.task
async def run_experiment(
    title: str,
    n_layer: int = 3,
    n_head: int = 4,
    n_embd: int = 128,
    dropout: float = 0.0,
    device_batch_size: int = 2,
    learning_rate: float = 3e-4,
    time_budget_sec: int = 45,
) -> dict:
    """Train one TinyGPT experiment and return its val_bpb (lower is better).

    Trains a causal language model on the climbmix corpus for ``time_budget_sec``
    using the given architecture / optimization knobs, then evaluates validation
    bits-per-byte. This task is memory-bound: peak RAM grows with
    ``device_batch_size * n_head`` (the attention matrix) and with
    ``n_layer * n_embd^2`` (parameters + activations). You do not need to think
    about compute — the runtime sizes each call and retries on OOM.

    Args:
        title: Short human-readable name for this experiment.
        n_layer: Number of transformer blocks (depth).
        n_head: Number of attention heads (must divide n_embd).
        n_embd: Embedding / hidden width.
        dropout: Dropout probability.
        device_batch_size: Sequences per step.
        learning_rate: AdamW learning rate.
        time_budget_sec: Wall-clock training budget in seconds.

    Returns:
        A dict with ``title``, ``val_bpb``, ``model_name``, ``n_params``,
        ``steps``, ``device`` and ``notes``.
    """
    bundle = await build_bundle()
    await materialize_cache(bundle)

    config = ExperimentConfig(
        title=title,
        n_layer=min(n_layer, MAX_N_LAYER),
        n_head=min(n_head, MAX_N_HEAD),
        n_embd=min(n_embd, MAX_N_EMBD),
        dropout=dropout,
        device_batch_size=min(device_batch_size, MAX_DEVICE_BATCH_SIZE),
        learning_rate=learning_rate,
        time_budget_sec=time_budget_sec,
    )
    if n_embd % config.n_head != 0:
        config = dataclasses.replace(config, n_embd=(config.n_embd // config.n_head) * config.n_head)
    import train

    result = train.run_training(config)
    return {
        "title": result.title,
        "val_bpb": round(result.val_bpb, 6),
        "model_name": result.model_name,
        "n_params": result.n_params,
        "steps": result.steps,
        "device": result.device,
        "notes": result.notes,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

INSTRUCTIONS = f"""\
You are a senior ML-engineer agent running autonomous research, in the spirit of
karpathy/autoresearch. Your goal: find a TinyGPT configuration that MINIMIZES
val_bpb (validation bits per byte; LOWER is better) on the climbmix corpus.

Available tools:
- inspect_dataset — understand the corpus before training (call once at the start).
- search_arxiv — optional literature search for architecture/optimizer ideas.
- validate_experiment_config — check a config and estimate parameter count before training.
- record_hypothesis — log what you expect before each experiment (use memory_key from directive).
- get_leaderboard / compare_experiments — review prior runs (memory_key from directive).
- run_experiment — train one config and return val_bpb (the expensive step).

How to work:
1. Call inspect_dataset, then optionally search_arxiv for relevant ideas.
2. For each experiment: validate_experiment_config, record_hypothesis, then run_experiment.
3. Read the returned val_bpb, then propose the NEXT experiment that you think will
   improve it: vary depth/width/heads/batch/learning_rate one or two at a time and
   reason explicitly about why.
4. Keep the best config in mind; discard regressions. Use get_leaderboard or
   compare_experiments to analyze trends.
5. Run experiments until you've tried the number the user asks for, then STOP and
   reply with a short plain-text summary: the best config, its val_bpb, and what
   you learned.

Workshop limits (do not exceed): n_layer<={MAX_N_LAYER}, n_embd<={MAX_N_EMBD},
n_head<={MAX_N_HEAD}, device_batch_size<={MAX_DEVICE_BATCH_SIZE}. Prefer small
changes from the baseline (3 layers, 128 embd, batch 2). Sequence length is fixed at 512.

You do NOT need to think about compute, memory, or OOM — the runtime right-sizes
every training run for you and automatically retries with more memory if needed.
If your prior transcript already contains experiments, continue from them rather
than repeating work.
"""

DEFAULT_MAX_TURNS = 500


def build_agent(*, max_turns: int = DEFAULT_MAX_TURNS) -> Agent:
    """Construct the MLE autoresearch agent with a configurable turn budget."""
    return Agent(
        name="mle-autoresearch-agent",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[
            search_arxiv,
            inspect_dataset,
            validate_experiment_config,
            record_hypothesis,
            get_leaderboard,
            compare_experiments,
            run_experiment,
        ],
        max_turns=max_turns,
        call_llm=call_llm,
    )


agent = build_agent()


# ---------------------------------------------------------------------------
# Parent task: build bundle, run agent with memory, stream reports
# ---------------------------------------------------------------------------


@agent_env.task(report=True)
async def mle_autoresearch_agent(
    n_experiments: int = 4,
    num_shards: int = DEFAULT_NUM_SHARDS,
    memory_key: str = MEMORY_KEY,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AutoresearchOutput:
    """Drive the MLE autoresearch agent with durable memory and live reports."""
    bundle = await build_bundle(num_shards=num_shards)
    profile = await profile_bundle(bundle)

    # 1) Memory: resume the agent's transcript + leaderboard across runs.
    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    persisted = await memory.read_json.aio("memory/leaderboard.json", default=[])
    flyte.logger.info("Restored %d prior messages + %d prior experiments.", len(memory.messages), len(persisted))

    # 2) Reports: stream the agent's activity into a live tab.
    events: list[dict[str, Any]] = []

    async def on_event(ev) -> None:
        events.append({"type": ev.type, "data": ev.data})
        if ev.type in ("tool_start", "tool_end", "tool_error", "turn_start", "agent_end"):
            tab = flyte.report.get_tab("Activity")
            tab.replace(render_activity_log(events))
            await flyte.report.flush.aio()

    directive_text = directive(n_experiments, profile, persisted, memory_key)

    token = agent_progress_cb.set(on_event)
    run_agent = build_agent(max_turns=max_turns)
    try:
        result = await run_agent.run.aio(directive_text, memory=memory)
    finally:
        agent_progress_cb.reset(token)

    # 3) Build the leaderboard from the transcript and render the report tabs.
    leaderboard, best = parse_leaderboard(memory.messages)
    leaderboard_dicts = [dataclasses.asdict(e) for e in leaderboard]

    tab_lb = flyte.report.get_tab("Leaderboard")
    tab_lb.replace(render_leaderboard(leaderboard, best))

    # 4) Persist the leaderboard into memory + save, then render the memory panel.
    await memory.write_json.aio(
        "memory/leaderboard.json",
        leaderboard_dicts,
        actor="mle-autoresearch-agent",
        reason=f"leaderboard after {len(leaderboard)} experiments",
    )
    await memory.save.aio()
    audit = await memory.audit_tail(20)
    hypotheses = await memory.read_json.aio("memory/hypotheses.json", default=[])

    tab_mem = flyte.report.get_tab("Memory")
    tab_mem.replace(
        render_memory_panel(memory_key, len(memory.messages), leaderboard_dicts, audit, hypotheses)
    )

    await flyte.report.replace.aio(render_summary(directive_text, leaderboard, best, result.summary or result.error or ""))
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
        help="Path to the Flyte config (defaults to ~/.flyte/config.yaml).",
    )
    args = parser.parse_args()

    flyte.init_from_config(args.config, image_builder="remote")

    async def main() -> None:
        run = await flyte.with_runcontext(copy_style="all").run.aio(
            mle_autoresearch_agent,
            n_experiments=args.n_experiments,
            num_shards=args.num_shards,
            memory_key=args.memory_key,
            max_turns=args.max_turns,
        )
        print(f"View run at: {run.url}")

    asyncio.run(main())
