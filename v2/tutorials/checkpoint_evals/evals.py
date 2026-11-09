# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# ///
"""Evaluate every checkpoint: an eval suite that runs itself whenever a checkpoint is published.

    new version of math-sft-checkpoint ──(OnArtifact trigger)──> evaluate_checkpoint
        ├─ run_generate  gsm8k     (L4 GPU, vLLM)
        ├─ run_choice    arc_easy  (L4 GPU, log-likelihood)
        └─ run_choice    mmlu      (L4 GPU, log-likelihood, the regression guard)
      -> publish scores as a new version of math-sft-checkpoint-evals

The evaluator knows nothing about the trainer. It is deployed once, and its trigger launches one
eval run per new checkpoint version, while training keeps going. Each benchmark result is cached
per (checkpoint, benchmark, benchmark version), so re-evaluating old checkpoints with a new
benchmark only runs the new one.

    python evals.py                                   # deploy the evaluator and its trigger
    python evals.py --backfill <train_run> arc_challenge   # add a benchmark to every checkpoint
                                                           # (register it in config.BENCHMARKS first)
"""

import asyncio
import json
import sys
import time

import flyte
import flyte.artifacts as artifacts
import flyte.report
from benchmarks import load_items, score_choice, score_generate
from config import (
    BENCHMARKS,
    CHECKPOINT_ARTIFACT,
    DEFAULT_SUITE,
    RESULTS_ARTIFACT,
    SMOKE_LIMIT,
    driver_image,
    gpu_image,
)
from flyte.io import Dir, File
from reports import scorecard_html

# {{docs-fragment envs}}
# Generative evals: one L4 per (checkpoint, benchmark). vLLM runs in the task's process, and
# uses PyTorch's sampler instead of FlashInfer's, which would JIT-compile with nvcc at startup.
generate_env = flyte.TaskEnvironment(
    name="ckpt-evals-generate",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu="L4:1", disk="40Gi"),
    env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
              "TOKENIZERS_PARALLELISM": "false"},
)

# Multiple-choice evals score every answer option's log-likelihood, one L4 per benchmark.
choice_env = flyte.TaskEnvironment(
    name="ckpt-evals-choice",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu="L4:1", disk="40Gi"),
    env_vars={"TOKENIZERS_PARALLELISM": "false"},
)

eval_env = flyte.TaskEnvironment(
    name="ckpt-evals",
    image=driver_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[generate_env, choice_env],
)

# The cache key is the task inputs (checkpoint, benchmark, benchmark version, limit) under one
# explicit cache version, so editing a report does not throw away finished evals.
EVAL_CACHE = flyte.Cache("override", version_override="checkpoint-evals-1")
# {{/docs-fragment envs}}


# {{docs-fragment benchmark_tasks}}
@generate_env.task(cache=EVAL_CACHE, retries=1)
def run_generate(checkpoint: Dir, benchmark: str, version: str, limit: int) -> str:
    """Score one checkpoint on one generative benchmark."""
    model_dir = checkpoint.download_sync()
    return json.dumps(score_generate(model_dir, load_items(benchmark, limit, version), version))


@choice_env.task(cache=EVAL_CACHE, retries=1)
def run_choice(checkpoint: Dir, benchmark: str, version: str, limit: int) -> str:
    """Score one checkpoint on one multiple-choice benchmark."""
    model_dir = checkpoint.download_sync()
    return json.dumps(score_choice(model_dir, load_items(benchmark, limit, version), benchmark, version))
# {{/docs-fragment benchmark_tasks}}


# {{docs-fragment trigger}}
on_checkpoint = flyte.Trigger(
    name="eval_new_checkpoint",
    description="Run the eval suite on every new version of the math SFT checkpoint",
    automation=flyte.OnArtifact(name=CHECKPOINT_ARTIFACT),
    inputs={
        "checkpoint": flyte.TriggeredArtifact,
        "step": flyte.TriggeredPartition("step"),
        "train_run": flyte.TriggeredPartition("train_run"),
        "profile": flyte.TriggeredPartition("profile"),
    },
)
# {{/docs-fragment trigger}}


# {{docs-fragment evaluate}}
@eval_env.task(triggers=[on_checkpoint], report=True, produces_artifacts=True)
async def evaluate_checkpoint(
    checkpoint: Dir, step: str, train_run: str, profile: str = "full", suite: list[str] = DEFAULT_SUITE
) -> File:
    """Fan the suite out over one checkpoint and publish the scores as an artifact version."""
    started = time.time()
    calls = []
    for name in suite:
        spec = BENCHMARKS[name]
        limit = SMOKE_LIMIT if profile == "smoke" else spec["limit"]
        task = run_generate if spec["kind"] == "generate" else run_choice
        calls.append(task.aio(checkpoint=checkpoint, benchmark=name, version=spec["version"], limit=limit))
    scores = {s["benchmark"]: s for s in map(json.loads, await asyncio.gather(*calls))}

    result = {"train_run": train_run, "step": int(step), "profile": profile, "checkpoint": checkpoint.path,
              "eval_run": flyte.ctx().action.run_name, "started_at": started, "finished_at": time.time(),
              "scores": scores}
    await flyte.report.replace.aio(scorecard_html(result), do_flush=True)

    path = "/tmp/scores.json"
    with open(path, "w") as f:
        json.dump(result, f)
    summary = ", ".join(f"{k} {v['accuracy']:.1f}" for k, v in scores.items())
    return artifacts.new(await File.from_local(path), artifacts.Metadata(
        name=RESULTS_ARTIFACT,
        description=f"step {int(step)} of {train_run}: {summary}",
        partitions={"train_run": train_run, "step": step, "profile": profile},
        attrs={k: f"{v['accuracy']:.2f}" for k, v in scores.items()},
    ))
# {{/docs-fragment evaluate}}


# {{docs-fragment backfill}}
def backfill(train_run: str, extra: list[str]) -> list[str]:
    """Re-run the suite plus ``extra`` benchmarks on every checkpoint of one training run.

    Benchmarks that already ran on a checkpoint come back from the cache; only ``extra`` runs.
    """
    from flyte.remote import Artifact

    unknown = [b for b in extra if b not in BENCHMARKS]
    if unknown:
        raise ValueError(f"add {unknown} to BENCHMARKS in config.py before backfilling")
    suite = DEFAULT_SUITE + [b for b in extra if b not in DEFAULT_SUITE]
    urls = []
    for art in Artifact.listall(name=CHECKPOINT_ARTIFACT, train_run=train_run, latest_per_partition=True):
        p = art.partitions
        run = flyte.run(evaluate_checkpoint, checkpoint=art, step=p["step"], train_run=train_run,
                        profile=p["profile"], suite=suite)
        urls.append(run.url)
    return urls
# {{/docs-fragment backfill}}


if __name__ == "__main__":
    flyte.init_from_config()
    if "--backfill" in sys.argv:
        i = sys.argv.index("--backfill")
        for url in backfill(sys.argv[i + 1], sys.argv[i + 2:]):
            print(url)
    else:
        # Deploying registers the tasks and activates the trigger.
        flyte.deploy(eval_env)
