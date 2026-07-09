# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.5.4",
#    "litellm",
#    "httpx",
#    "pydantic-monty",
#    "unionai-sandbox[flyte]",
#    "torch",
#    "numpy",
#    "pyarrow",
#    "requests",
#    "tiktoken",
#    "rustbpe",
# ]
# main = "parallelized_autoresearch"
# params = "--n-experiments 6 --batch-size 3 --num-shards 1"
# ///
"""Parallelized autoresearch agent — code-mode MLE agent with batched sandbox experiments."""

from __future__ import annotations

import dataclasses
from typing import Any

import flyte
import flyte.errors
import flyte.report
from flyte.ai.agents import Agent, MemoryStore, agent_progress_cb, tool

from autoresearch_types import AutoresearchOutput, DEFAULT_MAX_STEPS, DEFAULT_NUM_SHARDS, MAX_DEVICE_BATCH_SIZE, MAX_N_EMBD, MAX_N_HEAD, MAX_N_LAYER
from bundle import agent_env, build_bundle, experiment_env, materialize_cache, profile_bundle
import tools
import ui

MODEL = "claude-sonnet-4-6"
MAX_OOM_RETRIES = 3


async def _run_experiment_body(
    title: str,
    time_budget_sec: int,
    memory_key: str,
) -> dict:
    """Execute one sandbox training run (no OOM retry — used as a mapped sub-task)."""
    train_py = await tools.load_train_code(memory_key, title)
    config_overrides = await tools.load_config_overrides(memory_key, title)
    duplicate = await tools.check_duplicate_config(memory_key, title, train_py, config_overrides)
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
        await tools.record_experiment_result(
            memory_key,
            result,
            actor="parallelized-autoresearch",
        )
        return result
    bundle = await build_bundle()
    cache_dir = await materialize_cache(bundle)
    result = await tools.run_train_in_sandbox(
        cache_dir,
        train_py,
        title=title,
        time_budget_sec=time_budget_sec,
        config_overrides=config_overrides or None,
    )
    if result.get("success"):
        await tools.record_promising_run(memory_key, title, result)
        await tools.register_config_signature(
            memory_key,
            title,
            train_py,
            config_overrides,
            actor="parallelized-autoresearch",
        )
    await tools.record_experiment_result(
        memory_key,
        result,
        actor="parallelized-autoresearch",
    )
    return result


@experiment_env.task
async def run_experiment_body(
    title: str,
    time_budget_sec: int = 45,
    memory_key: str = tools.MEMORY_KEY_FANOUT,
) -> dict:
    """Run one edited ``train.py`` inside a sandbox (mapped sub-task)."""
    return await _run_experiment_body(title, time_budget_sec, memory_key)


@experiment_env.task
async def run_experiment(
    title: str,
    time_budget_sec: int = 45,
    memory_key: str = tools.MEMORY_KEY_FANOUT,
) -> dict:
    """Train using agent-edited ``train.py`` with platform OOM self-healing.

    Safe to call directly or via ``flyte_map("run_experiment", titles, ...)``.
    """
    resources = tools.RESOURCE_FLOOR
    attempt = 0
    while True:
        try:
            result = await run_experiment_body.override(resources=resources).aio(
                title=title,
                time_budget_sec=time_budget_sec,
                memory_key=memory_key,
            )
        except flyte.errors.OOMError:
            if attempt >= MAX_OOM_RETRIES:
                raise
            resources = tools.bump_memory(resources)
            attempt += 1
            flyte.logger.warning(
                "run_experiment Flyte OOM for %s; retry memory=%s",
                title,
                resources.memory,
            )
            continue

        if isinstance(result, dict):
            result["resources"] = f"cpu={resources.cpu}, mem={resources.memory}"
            result["oom_retries"] = attempt

        if isinstance(result, dict) and result.get("oom"):
            if attempt >= MAX_OOM_RETRIES:
                return result
            resources = tools.bump_memory(resources)
            attempt += 1
            flyte.logger.warning(
                "run_experiment sandbox OOM for %s; retry memory=%s",
                title,
                resources.memory,
            )
            continue

        return result


@tool
@agent_env.task
async def run_experiment_batch(
    titles: list[str],
    time_budget_sec: int = 45,
    memory_key: str = tools.MEMORY_KEY_FANOUT,
    concurrency: int = 4,
    batch_id: str = "",
) -> dict:
    """Run multiple ``run_experiment`` calls in parallel via ``flyte.map``.

    Prefer this over hand-rolling ``flyte_map`` when you already have a list of
    experiment titles with saved ``train.py`` edits.

    Args:
        titles: Experiment titles whose code was saved with ``edit_train_code`` or
        ``edit_train_code_batch``.
        time_budget_sec: Wall-clock budget passed to each run.
        memory_key: Memory namespace from your directive.
        concurrency: Max parallel sandbox runs (default 4).
        batch_id: Optional label attached to the returned batch metadata.

    Returns:
        A dict with ``batch_size``, ``titles``, ``results``, ``evaluation``,
        and ``seed`` (platform default parent for the next batch).
    """
    group = batch_id or f"batch-{len(titles)}"
    payload = await tools.run_experiment_batch_impl(
        run_experiment,
        titles,
        time_budget_sec=time_budget_sec,
        memory_key=memory_key,
        concurrency=concurrency,
        group_name=group,
    )
    evaluation, seed = await tools.finalize_batch_results(
        memory_key,
        payload["results"],
        batch_id=batch_id or group,
    )
    payload["evaluation"] = evaluation
    payload["seed"] = seed
    return payload


INSTRUCTIONS = f"""\
You are a senior ML-engineer agent running karpathy/autoresearch-style research by
**editing train.py** and **batching parallel experiments**. Your goal: MINIMIZE
val_bpb (LOWER is better).

You operate in CODE MODE. Each turn, write ONE ```python``` block that calls the
available functions, OR reply in plain text when finished. The last expression in
your code block is returned as the observation.

Core tools (same as the single-threaded code-edit agent):
- get_code_edit_history — **call first on resumed sessions**: prior edits, val_bpb, vs-best deltas
- get_baseline_train_code, edit_train_code, edit_train_code_batch, read_train_code, get_promising_code, get_seed_train_code
- inspect_dataset, search_arxiv
- record_hypothesis, get_leaderboard, compare_experiments
- run_experiment — one sandbox training run (OOM-healed by the platform)

Saving edits (required for visible diffs and distinct runs):
- **Batch 1 only:** you may use ``config_overrides`` for a quick architecture/LR sweep via
  ``edit_train_code_batch(edits=[{{"title": "...", "config_overrides": {{"n_layer": 6}}, "change_summary": "..."}}])``.
- **Batch 2 and later:** every edit must include a **substantive ``train_py`` change**
  (learning-rate schedule, optimizer/weight_decay, grad clipping, warmup, etc.).
  ``config_overrides`` alone is **rejected** after the first batch.
- After each ``run_experiment_batch``, the platform sets the **seed** to the batch best
  (when it beats the prior seed). The next batch forks from that seed automatically;
  use ``get_seed_train_code()`` or ``read_train_code(seed_title)`` to inspect it.
- Optional ``parent_title`` overrides the platform seed when forking.
- Do **not** save baseline ``train.py`` without overrides — the platform rejects identical edits.
- Duplicate configs (same effective train.py + overrides) are rejected at run time.
- ``config_overrides`` fields: ``n_layer``, ``n_head``, ``n_embd``, ``dropout``,
  ``device_batch_size``, ``learning_rate``, ``time_budget_sec``, ``max_steps``.

Training budget (fair comparison across architectures):
- Default **max_steps={DEFAULT_MAX_STEPS}** with **time_budget_sec=45** as a safety cap.
  All models train for the same step count unless they hit the wall-clock limit.
- Check ``steps`` in batch results — if a run stopped early on time, the model may be too large.

Batch / fan-out tools:
- record_batch_plan(batch_id, experiments) — persist a multi-experiment plan
- get_batch_plan(batch_id) — reload a plan
- record_batch_hypotheses(experiments) — write hypotheses for every title in a batch
- edit_train_code_batch(edits) — save all ``train.py`` edits in one memory transaction
- run_experiment_batch(titles, concurrency=...) — parallel ``flyte.map`` over runs
- evaluate_batch_results(results, batch_id=...) — rank successes vs failures

Parallel fan-out in code:
- After saving edits, you may call ``run_experiment_batch(titles, ...)`` OR
  ``flyte_map("run_experiment", titles, budgets, keys, concurrency=N)`` where
  budgets/keys are lists matching titles.

Typical batch loop (aim for **≤8 code turns** before your plain-text summary):
0. If prior research exists in your directive, ``get_code_edit_history()`` then
   ``get_seed_train_code()`` / ``read_train_code(seed_title)`` before planning new batches.
1. Turn 1: ``get_baseline_train_code()`` + ``inspect_dataset()``.
2. Turn 2: ``record_batch_plan`` then ``edit_train_code_batch(edits=[...])`` for the whole batch.
3. Turn 3: ``record_batch_hypotheses`` + ``run_experiment_batch(titles, concurrency=...)``.
   Check ``seed`` in the result — the platform uses it as the parent for batch 2+ edits.
4. Turn 4+: edit ``train.py`` from the platform seed for the next batch, or reply in plain text when done.

Batch diversity (required):
- Every title in a batch must test a **distinct hypothesis** — no duplicate configs or renames.
- **Spread axes across the batch**: e.g. one edit tweaks depth/width, another changes the
  **training loop** (cosine LR, AdamW betas, weight decay), another regularization or batch size.
- Avoid LR micro-sweeps (±30% of the current best LR) after batch 1 — those rarely beat a plateau.
- Vary **one or two knobs per edit**; state the change in ``change_summary`` and
  ``record_batch_hypotheses``.
- Use ``evaluate_batch_results`` to see **which axis** helped, then explore under-tested axes.

Plateau rule (required):
- If **3 consecutive batches** fail to beat the global best val_bpb by more than **0.01**,
  stop hyperparameter micro-sweeps. Switch to **training-loop code edits** in ``train.py``
  (scheduler, optimizer, regularization, data/loss changes).

Rules:
- Prefer ``edit_train_code_batch`` over repeated ``edit_train_code`` when saving 2+ titles.
- Every edit must keep ``run_training(config: ExperimentConfig) -> ExperimentResult``.
- Do NOT size compute — the platform right-sizes and retries OOM per run.
- Workshop limits: n_layer<={MAX_N_LAYER}, n_embd<={MAX_N_EMBD}, n_head<={MAX_N_HEAD},
  device_batch_size<={MAX_DEVICE_BATCH_SIZE}, seq_len=512.
- Prefer ``run_experiment_batch`` over hand-written ``flyte_map`` unless you need it.
- Monty sandbox: no imports, no dict mutation, no augmented assignment (`+=`).
- **Always finish with plain text (no code block)** once you have results to report.
"""

DEFAULT_MAX_TURNS = 50


def build_fanout_agent(*, max_turns: int = DEFAULT_MAX_TURNS) -> Agent:
    """Construct the fan-out agent (``code_mode=True``) with a configurable turn budget."""
    return Agent(
        name="parallelized-autoresearch",
        instructions=INSTRUCTIONS,
        model=MODEL,
        tools=[
            tools.search_arxiv,
            tools.inspect_dataset,
            tools.get_baseline_train_code,
            tools.get_code_edit_history,
            tools.edit_train_code,
            tools.edit_train_code_batch,
            tools.read_train_code,
            tools.get_promising_code,
            tools.get_seed_train_code,
            tools.record_hypothesis,
            tools.get_leaderboard,
            tools.compare_experiments,
            tools.record_batch_plan,
            tools.get_batch_plan,
            tools.record_batch_hypotheses,
            run_experiment,
            run_experiment_batch,
            tools.evaluate_batch_results,
        ],
        code_mode=True,
        max_turns=max_turns,
        call_llm=tools.call_llm,
    )


agent = build_fanout_agent()


# {{docs-fragment agent}}
@agent_env.task(report=True)
async def parallelized_autoresearch(
    n_experiments: int = 6,
    num_shards: int = DEFAULT_NUM_SHARDS,
    memory_key: str = tools.MEMORY_KEY_FANOUT,
    batch_size: int = 3,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AutoresearchOutput:
    """Drive the fan-out code-edit MLE agent with sandbox batch execution."""
    bundle = await build_bundle(num_shards=num_shards)
    profile = await profile_bundle(bundle)

    memory = await MemoryStore.get_or_create.aio(key=memory_key)
    persisted = await memory.read_json.aio("memory/leaderboard.json", default=[])
    promising = await memory.read_json.aio("memory/promising_code.json", default=[])
    history = await tools.load_research_history(memory_key)
    seed = await tools.load_seed_train_code(memory_key)
    flyte.logger.info(
        "Fan-out agent restored %d messages, %d experiments, %d promising edits, best val_bpb=%s, seed=%s.",
        len(memory.messages),
        len(persisted),
        len(promising),
        history.get("best_val_bpb"),
        seed.get("title") if seed else None,
    )

    events: list[dict[str, Any]] = []

    async def on_event(ev) -> None:
        events.append({"type": ev.type, "data": ev.data})
        if ev.type in ("tool_start", "tool_end", "tool_error", "turn_start", "agent_end"):
            tab = flyte.report.get_tab("Activity")
            tab.replace(ui.render_activity_log(events))
            await flyte.report.flush.aio()
        if ev.type == "tool_end" and ev.data.get("tool") in (
            "edit_train_code",
            "edit_train_code_batch",
            "<sandbox>",
        ):
            edits = await tools.load_saved_code_edits(memory_key)
            if edits:
                flyte.report.get_tab("Code edits").replace(ui.render_code_edits_panel(edits))
                await flyte.report.flush.aio()

    directive_text = ui.directive_code_edit_fanout(
        n_experiments,
        profile,
        memory_key,
        batch_size=batch_size,
        history=history,
    )

    token = agent_progress_cb.set(on_event)
    run_agent = build_fanout_agent(max_turns=max_turns)
    try:
        result = await run_agent.run.aio(directive_text, memory=memory)
    finally:
        agent_progress_cb.reset(token)

    leaderboard, best = ui.parse_leaderboard(
        memory.messages,
        promising_fallback=promising,
    )
    leaderboard_dicts = [dataclasses.asdict(e) for e in leaderboard]
    code_edits = await tools.load_saved_code_edits(memory_key)

    tab_lb = flyte.report.get_tab("Leaderboard")
    tab_lb.replace(ui.render_leaderboard(leaderboard, best))

    flyte.report.get_tab("Code edits").replace(
        ui.render_code_edits_panel(code_edits, best_title=best.title if best else None)
    )

    await memory.write_json.aio(
        "memory/leaderboard.json",
        leaderboard_dicts,
        actor="parallelized-autoresearch",
        reason=f"leaderboard after {len(leaderboard)} experiments",
    )
    await memory.save.aio()
    audit = await memory.audit_tail(20)
    hypotheses = await memory.read_json.aio("memory/hypotheses.json", default=[])
    promising = await memory.read_json.aio("memory/promising_code.json", default=[])

    tab_mem = flyte.report.get_tab("Memory")
    tab_mem.replace(
        ui.render_memory_panel(
            memory_key,
            len(memory.messages),
            leaderboard_dicts,
            audit,
            hypotheses,
            persisted_promising=promising,
            code_edits=code_edits,
        )
    )

    summary_body = result.summary or result.error or ""
    if result.error and leaderboard:
        best_line = f" Best val_bpb so far: {best.val_bpb} ({best.title})." if best and best.val_bpb else ""
        summary_body = f"{result.error}{best_line}"

    await flyte.report.replace.aio(
        ui.render_summary(
            directive_text,
            leaderboard,
            best,
            summary_body,
            code_edits=code_edits,
        )
    )
    await flyte.report.flush.aio()

    return AutoresearchOutput(
        directive=directive_text,
        dataset_profile=profile,
        best=best,
        leaderboard=leaderboard,
        summary=summary_body,
        memory_key=memory_key,
        total_experiments=len(leaderboard),
    )


# {{/docs-fragment agent}}


# {{docs-fragment main}}
if __name__ == "__main__":
    import argparse
    import asyncio
    import os

    parser = argparse.ArgumentParser(description="Parallelized autoresearch agent (CODE MODE)")
    parser.add_argument("--n-experiments", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--memory-key", default=tools.MEMORY_KEY_FANOUT)
    parser.add_argument(
        "--config",
        default=os.environ.get("FLYTE_CONFIG", os.path.expanduser("~/.flyte/config.yaml")),
    )
    args = parser.parse_args()

    flyte.init_from_config(args.config, image_builder="remote")

    async def main() -> None:
        run = await flyte.with_runcontext(copy_style="all").run.aio(
            parallelized_autoresearch,
            n_experiments=args.n_experiments,
            num_shards=args.num_shards,
            memory_key=args.memory_key,
            batch_size=args.batch_size,
            max_turns=args.max_turns,
        )
        print(f"View run at: {run.url}")

    asyncio.run(main())
# {{/docs-fragment main}}
