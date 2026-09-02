# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.4.0",
#    "unionai-reuse>=0.1.3",
#    "async-lru>=2.0.0",
#    "torch>=2.5.0",
#    "transformers>=4.51.0",
#    "peft>=0.13.0",
#    "vllm>=0.11.0",
#    "accelerate>=0.34.0",
#    "datasets>=3.0.0",
# ]
# main = "train_rl_clustered"
# params = ""
# ///
"""
Part 2: Async multi-node GRPO — the trainer on a ClusteredTaskEnvironment
=========================================================================

This module extends the Part-1 tutorial (``rl_grpo_lora.py``, imported here for the rollout /
reward / eval building blocks) with the Part-2 ideas:

1. **The GRPO trainer becomes a multi-node clustered task.** ``train_step_clustered`` runs on a
   ``ClusteredTaskEnvironment``: `replicas` pods launched as ONE Kubernetes JobSet, bootstrapped
   by torchrun, cooperating via ``torch.distributed``. Same call shape as Part 1's ``train_step``
   — the multi-node part is an *environment declaration*, not a rewrite.

2. **Generation pipelines against training (async, one-step off-policy).** While the clustered
   trainer spins up and trains on iteration N's rollouts, the warm vLLM pool is already
   generating iteration N+1's rollouts with the current (one-step-stale) adapter. The JobSet
   cold start disappears under generation the pool was doing anyway. ``pipelined=False`` runs
   the same loop strictly sequentially so the two can be measured against each other — the
   report's timing chart is the receipt.

3. **The real GRPO objective.** Because pipelining trains on rollouts sampled by the *previous*
   adapter, the plain REINFORCE loss of Part 1 is no longer sound. This trainer implements the
   token-level clipped-importance-ratio surrogate (ratio from vLLM's sampling-time logprobs
   carried in each ``Rollout``) plus a k3 KL penalty against the adapter-disabled base model —
   i.e. GRPO as published, not a simplification. Optimizer (AdamW) state persists across
   iterations through the object store, exactly like the adapter.

4. **A held-out eval.** ``evaluate`` (Part 1) scores a fixed question set with greedy decoding
   on the warm pool — once for the raw base model (the baseline) and again after every training
   step. The live report charts the delta.

What users learn here about clustered task environments:
- how to CREATE one (``ClusteredTaskEnvironment(replicas=..., nproc_per_node=..., runtime=
  TorchRun(), failure_policy=ClusterFailurePolicy(...))``),
- how to USE one (decorate a plain async function with ``@train_env.task``; call it with
  ``await`` from a driver exactly like any other task),
- what runs inside (every rank executes the same function; torchrun has already set
  RANK/WORLD_SIZE/MASTER_ADDR; ``flyte.ctx()`` exposes rank/world_size/node_rank),
- the two hard rules: rank 0's return value is the task output (other ranks' are discarded, but
  every rank must return matching types), and a clustered task has NO controller — it cannot
  call other tasks, so the loop stays in the plain-env driver,
- the one DDP discipline multi-node training demands: **every rank must execute the same number
  of collective operations**. Per-sample ``backward()`` with data-dependent skips desyncs NCCL
  the moment ranks disagree; see ``ddp_shard_step`` for the pattern that makes the invariant
  hold by construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import flyte
import flyte.io
import flyte.report
from flyte.clustered import ClusteredTaskEnvironment, ClusterFailurePolicy, TorchRun
from pydantic import BaseModel

# Part 1 building blocks reused verbatim: image, warm vLLM pool, reward, eval, adapter init.
from report_helpers import IterationMetrics, render_report
from rl_grpo_lora import (
    _PROFILE_ENV_VARS,
    BASE_MODEL_REPO,
    DATASET,
    DEFAULT_DATASET,
    EVAL_SET_SIZE,
    GPU_TYPE,
    GROUP_SIZE,
    HF_SECRET,
    LEARNING_RATE,
    LORA_RANK,
    NUM_ITERATIONS,
    PROFILE,
    PROFILE_NAME,
    PROMPTS_PER_ITER,
    Rollout,
    _extract_answer,
    _group_normalized_advantages,
    evaluate,
    generate,
    image,
    init_adapter,
    reward_env,
    rollout_env,
    score_group,
)
from rl_grpo_lora import train_env as part1_train_env  # init_adapter lives here → must register it

logger = logging.getLogger(__name__)

# --- Part-2 knobs (sized by GRPO_PROFILE; see PROFILES in rl_grpo_lora.py) ----------------------
REPLICAS: int = PROFILE["train_replicas"]  # pods (== nodes) in the trainer JobSet
NPROC_PER_NODE: int = PROFILE["train_nproc_per_node"]  # ranks per pod (== GPUs per pod)
#   world_size = REPLICAS * NPROC_PER_NODE
USE_GPU = True
_BACKEND = "nccl" if USE_GPU else "gloo"

CLIP_EPS = 0.2  # PPO/GRPO clip window: ratios outside [1-eps, 1+eps] stop contributing gradient
KL_BETA = 0.03  # weight of the KL(policy ‖ adapter-disabled base) penalty
TRAIN_EPOCHS_PER_STEP: int = PROFILE["train_epochs_per_step"]  # optimizer steps per clustered call;
#   >1 amortizes the JobSet cold start and is legitimate *because* the objective clips — each extra
#   epoch re-computes ratios against the just-updated policy, PPO-style.

_train_resources = (
    flyte.Resources(
        cpu=PROFILE["train_cpu"],
        memory=PROFILE["train_memory"],
        gpu=flyte.GPU(GPU_TYPE, PROFILE["train_gpus"]),  # per POD; must be >= nproc_per_node
        shm="auto",
    )
    if USE_GPU
    else flyte.Resources(cpu=(2, 4), memory=("4Gi", "8Gi"))
)

# ----------------------------------------------------------------------------------------------------
# THE clustered task environment — this is the new concept Part 2 teaches.
# ----------------------------------------------------------------------------------------------------
# Each call to a task decorated with this env emits ONE Kubernetes JobSet of `replicas` pods.
# Every pod runs torchrun with `nproc_per_node` workers; all workers rendezvous into a single
# torch.distributed process group. `failure_policy(max_restarts=2)` = if ANY pod dies, the WHOLE
# set restarts together (up to twice) — partial restarts would leave ranks desynchronized.
# NOTE: `reusable` is not supported here (clustered = run-to-completion JobSet, not a warm actor);
# the warm pool stays where it belongs — on the Part-1 rollout env.
# {{docs-fragment clustered_env}}
train_env = ClusteredTaskEnvironment(
    name="rl-grpo-train-cluster",
    image=image,  # same image as Part 1 — torch/peft/transformers already in it
    resources=_train_resources,  # per-POD resources (not per-JobSet)
    replicas=REPLICAS,
    nproc_per_node=NPROC_PER_NODE,
    runtime=TorchRun(rdzv_backend="static", max_restarts=0),  # in-pod torchrun; restarts are JobSet-level
    failure_policy=ClusterFailurePolicy(max_restarts=2),
    secrets=[HF_SECRET],
    env_vars=_PROFILE_ENV_VARS,
)
# {{/docs-fragment clustered_env}}


class TrainMetrics(BaseModel):
    """Cross-rank-reduced training statistics returned alongside the new adapter."""

    mean_loss: float
    mean_ratio: float  # mean importance ratio pi_new/pi_old — drifts off 1.0 when pipelining
    clip_fraction: float  # fraction of tokens whose ratio left the [1-eps, 1+eps] window
    mean_kl: float  # mean k3 KL(policy ‖ adapter-disabled base)
    grad_norm: float
    contributing: int  # samples with non-zero advantage (metric only — all valid samples train)
    num_valid: int  # samples carrying token_ids + logprobs that entered the loss


# ----------------------------------------------------------------------------------------------------
# The GRPO objective and the DDP collective pattern — pure functions so `test_grpo_loss_ddp.py`
# can exercise them on CPU without Flyte or a cluster.
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment grpo_loss}}
def grpo_sample_loss(
    new_logp: Any,
    old_logp: Any,
    ref_logp: Any,
    advantage: float,
    clip_eps: float = CLIP_EPS,
    kl_beta: float = KL_BETA,
) -> tuple[Any, dict[str, float]]:
    """Token-level clipped-ratio GRPO loss for ONE completion.

    All inputs are fp32 tensors of shape (T,) over the completion tokens; only ``new_logp``
    requires grad. ``old_logp`` is the *behavior* policy — vLLM's sampling-time logprobs, i.e.
    the adapter version that actually generated the rollout (one step stale when pipelining).
    ``ref_logp`` is the frozen base with the adapter disabled, for the KL anchor.

    loss = −( mean_t min(r_t·Â, clip(r_t, 1±ε)·Â) − β·mean_t KL_k3 ),  r_t = exp(new−old).

    The k3 estimator ``exp(δ) − δ − 1`` (δ = ref − new) is non-negative and low-variance
    (Schulman). At r = 1 the surrogate's *gradient* equals REINFORCE's (its value does not).
    """
    import torch

    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage).mean()
    delta = ref_logp - new_logp
    kl = (torch.exp(delta) - delta - 1.0).mean()
    loss = -(surrogate - kl_beta * kl)
    stats = {
        "loss": float(loss.detach()),
        "ratio": float(ratio.mean().detach()),
        "kl": float(kl.detach()),
        "clip_frac": float(((ratio - 1.0).abs() > clip_eps).float().mean().detach()),
    }
    return loss, stats


def ddp_shard_step(ddp_model: Any, sample_loss_fns: list, sync_forward: Any) -> None:
    """Accumulate per-sample gradients, then synchronize with EXACTLY ONE collective backward.

    DDP fires its gradient all-reduce inside ``backward()``, so all ranks must execute the same
    number of collective-bearing backwards — but shards are data-dependent and may be empty on
    some ranks. The documented escape hatch: backwards inside ``no_sync()`` accumulate grads
    locally with NO collectives (and free each sample's graph immediately), and the first
    forward-backward AFTER the context syncs everything. Every rank therefore runs one dummy
    1-token forward + zero-scaled backward, unconditionally — the collective count is identical
    on every rank *by construction*, empty shards included.

    (A bare ``(0.0 * sum(p.sum())).backward()`` would NOT work: it never passes through
    ``DDP.forward``, so the reducer is not prepared for the iteration.)
    """
    with ddp_model.no_sync():
        for fn in sample_loss_fns:
            fn().backward()  # local grad accumulation; per-sample graph freed here
    (sync_forward().float().sum() * 0.0).backward()  # the ONE synchronizing backward
# {{/docs-fragment grpo_loss}}


# ----------------------------------------------------------------------------------------------------
# The multi-node GRPO step. Every rank runs this function; gradients are averaged by DDP; rank 0
# saves and uploads the new adapter + optimizer state and broadcasts their URIs so all ranks
# return identical, correctly-typed outputs (rank 0's return IS the task output).
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment train_step_clustered}}
@train_env.task
async def train_step_clustered(
    base: flyte.io.Dir,
    rollouts: list[Rollout],
    rewards: list[float],
    adapter: flyte.io.Dir,
    version: int,
    opt_state: flyte.io.Dir | None = None,
) -> tuple[flyte.io.Dir, flyte.io.Dir, TrainMetrics]:
    """One GRPO step across REPLICAS × NPROC_PER_NODE ranks (DDP over the LoRA adapter).

    DDP, not FSDP, at this scale: the frozen 0.6B base fits every GPU, and DDP syncs only
    parameters with requires_grad — i.e. the few-MB LoRA adapter — so cross-node traffic is
    megabytes per step, comfortable on the clustered env's TCP interconnect. The FSDP swap
    (shard the frozen base so bigger policies fit) is a later scale-up, marked below.
    """
    import tempfile
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM

    ctx = flyte.ctx()

    # Bind this rank to its local GPU BEFORE init_process_group so NCCL picks the right device.
    if _BACKEND == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(ctx.local_rank or 0)
        device = torch.device(f"cuda:{ctx.local_rank or 0}")
    else:
        device = torch.device("cpu")

    # torchrun already populated RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Short timeout so a
    # desynced collective surfaces as an error in minutes, not NCCL's 30-minute default.
    dist.init_process_group(backend=_BACKEND, timeout=timedelta(minutes=10))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    logger.info("[rank %d/%d] node %s/%s device=%s", rank, world_size, ctx.node_rank, ctx.nnodes, device)

    local_base: str = await base.download()  # NB: every rank downloads — exercises the bundle path
    local_adapter: str = await adapter.download()
    local_opt: str | None = await opt_state.download() if opt_state is not None else None

    # low_cpu_mem_usage: with nproc_per_node ranks per pod each loading the base, the default loader's
    # full-size CPU staging copy per rank is what blows the pod's memory request on an 8B.
    base_model: Any = AutoModelForCausalLM.from_pretrained(
        local_base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device)
    peft_model: Any = PeftModel.from_pretrained(base_model, local_adapter, is_trainable=True).to(device)
    # FSDP swap for bigger policies goes here: wrap `peft_model` in FSDP(auto_wrap transformer
    # blocks, use_orig_params=True) instead of DDP, and gather a full state dict before saving.
    model = DDP(peft_model, device_ids=[device.index] if device.type == "cuda" else None)
    model.train()

    # Optimizer state persists across GRPO iterations through the object store, like the adapter.
    # Every rank builds AdamW over the SAME param sequence and every rank loads the same state —
    # they all step() on identical post-allreduce grads, so their states must match too.
    # NOTE: a loaded state_dict's param_groups override the constructor lr — fine with a constant
    # LR; revisit if a schedule is ever added.
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer: Any = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    if local_opt is not None:
        sd = torch.load(os.path.join(local_opt, "optimizer.pt"), map_location="cpu", weights_only=True)
        optimizer.load_state_dict(sd)  # casts state tensors to the params' device

    # Advantages are computed globally (group normalization must see whole groups, which may span
    # shards). Validity is a property of the payload, so EVERY rank computes the same n_valid with
    # zero communication — the loss normalization below depends on that.
    advantages: list[float] = _group_normalized_advantages(rollouts, rewards)
    valid = [
        i for i, r in enumerate(rollouts) if len(r.token_ids) > 0 and len(r.logprobs) == len(r.token_ids)
    ]
    if not valid:
        raise ValueError(
            "No rollout carries token_ids/logprobs — these rollouts were generated by a pre-Part-2 "
            "generate(); re-run generation with the updated module."
        )
    n_valid = len(valid)
    # Contiguous shard of the *valid* indices; zero-advantage samples still train (the KL term
    # anchors them) — skipping them would bias the KL and re-create data-dependent divergence.
    shard = valid[rank * n_valid // world_size : (rank + 1) * n_valid // world_size]

    sums = {"loss": 0.0, "ratio": 0.0, "kl": 0.0, "clip_frac": 0.0}

    def make_sample_fn(i: int):
        r = rollouts[i]

        def fn() -> Any:
            ids = torch.tensor([r.prompt_token_ids + r.token_ids], dtype=torch.long, device=device)
            p, t = len(r.prompt_token_ids), len(r.token_ids)
            # logits[p-1 : p-1+t] predict exactly the t completion tokens. Slice BEFORE the fp32
            # upcast + log_softmax so the transient is (t, vocab), not (seq, vocab).
            new_logp = (
                model(ids).logits[0, p - 1 : p - 1 + t].float().log_softmax(-1)
                .gather(-1, ids[0, p:].unsqueeze(-1)).squeeze(-1)
            )
            # Reference = same weights with LoRA disabled — one extra forward, no second model.
            # Called on peft_model directly (not the DDP wrapper): no_grad forwards must not
            # touch the DDP reducer.
            with torch.no_grad(), peft_model.disable_adapter():
                ref_logp = (
                    peft_model(ids).logits[0, p - 1 : p - 1 + t].float().log_softmax(-1)
                    .gather(-1, ids[0, p:].unsqueeze(-1)).squeeze(-1)
                )
            old_logp = torch.tensor(r.logprobs, dtype=torch.float32, device=device)
            loss, stats = grpo_sample_loss(new_logp, old_logp, ref_logp, advantages[i])
            for k in sums:
                sums[k] += stats[k]
            # Scale so the DDP-averaged gradient equals the single-process mean over ALL valid
            # samples: avg_ranks(grad(Σ_shard (W/N)·l_i)) = (1/N)·Σ_all grad(l_i).
            return loss * (world_size / n_valid)

        return fn

    grad_norm = 0.0
    for _epoch in range(TRAIN_EPOCHS_PER_STEP):
        for k in sums:
            sums[k] = 0.0
        optimizer.zero_grad(set_to_none=True)
        ddp_shard_step(
            model,
            [make_sample_fn(i) for i in shard],
            lambda: model(torch.zeros((1, 1), dtype=torch.long, device=device)).logits,
        )
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()  # identical on every rank: grads were averaged by DDP

    # Reduce metrics across ranks so the driver sees GLOBAL statistics, not rank 0's shard.
    # (Stats are from the final epoch; contributing counts non-zero advantages, as a metric only.)
    shard_contributing = sum(1 for i in shard if advantages[i] != 0.0)
    stats_t = torch.tensor(
        [sums["loss"], sums["ratio"], sums["kl"], sums["clip_frac"], float(len(shard)), float(shard_contributing)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
    n_global = max(int(stats_t[4]), 1)
    metrics = TrainMetrics(
        mean_loss=float(stats_t[0]) / n_global,
        mean_ratio=float(stats_t[1]) / n_global,
        mean_kl=float(stats_t[2]) / n_global,
        clip_fraction=float(stats_t[3]) / n_global,
        grad_norm=grad_norm,
        contributing=int(stats_t[5]),
        num_valid=n_valid,
    )

    # Rank 0 saves + uploads adapter and optimizer state, then broadcasts the URIs so every rank
    # returns identical, correctly-typed outputs (the broadcast doubles as the pre-exit sync).
    uris: list[str | None] = [None, None]
    if rank == 0:
        adapter_dir = tempfile.mkdtemp(prefix=f"adapter-v{version}-")
        peft_model.save_pretrained(adapter_dir)  # PeftModel → adapter_config.json + safetensors only
        opt_dir = tempfile.mkdtemp(prefix=f"optstate-v{version}-")
        torch.save(optimizer.state_dict(), os.path.join(opt_dir, "optimizer.pt"))
        uris = [(await flyte.io.Dir.from_local(adapter_dir)).path, (await flyte.io.Dir.from_local(opt_dir)).path]
        logger.info(
            "GRPO step v%d: %d valid rollouts (%d contributing), loss %.4f, ratio %.3f, clip %.3f, KL %.4f",
            version, metrics.num_valid, metrics.contributing, metrics.mean_loss,
            metrics.mean_ratio, metrics.clip_fraction, metrics.mean_kl,
        )
    dist.broadcast_object_list(uris, src=0)
    new_adapter = flyte.io.Dir.from_existing_remote(uris[0])
    new_opt_state = flyte.io.Dir.from_existing_remote(uris[1])

    dist.barrier()
    dist.destroy_process_group()
    return new_adapter, new_opt_state, metrics
# {{/docs-fragment train_step_clustered}}


# ----------------------------------------------------------------------------------------------------
# Datasets — the toy arithmetic set (wiring smoke test) or GSM8K (a real learning signal).
# ----------------------------------------------------------------------------------------------------
def _load_pairs(dataset: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train_pairs, eval_pairs) of (question, gold-integer-answer-string).

    ``arithmetic``: Part 1's 8 hardcoded questions. There is no held-out split at that size —
    eval reuses the same 8, so its accuracy is a wiring check, not a generalization claim.
    ``gsm8k``: HF ``openai/gsm8k`` (public, no token needed); train split for rollouts, the first
    EVAL_SET_SIZE test items held out for eval. Gold answers keep GSM8K's '#### N' convention;
    commas stripped ("1,200" → "1200") to match `_extract_answer`.
    """
    if dataset == "arithmetic":
        return list(DATASET), list(DATASET)
    if dataset == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")

        def pair(ex: dict) -> tuple[str, str]:
            return ex["question"], ex["answer"].rsplit("####", 1)[1].strip().replace(",", "")

        train_pairs = [pair(ex) for ex in ds["train"]]
        eval_pairs = [pair(ex) for ex in ds["test"].select(range(EVAL_SET_SIZE))]
        return train_pairs, eval_pairs
    raise ValueError(f"unknown dataset {dataset!r} (expected 'arithmetic' or 'gsm8k')")


# ----------------------------------------------------------------------------------------------------
# The async (pipelined) driver. Plain env — clustered tasks cannot call subtasks, so the loop
# lives here. depends_on registers every environment the driver's tasks come from.
# ----------------------------------------------------------------------------------------------------
driver_env = flyte.TaskEnvironment(
    name="rl-grpo-driver-v2",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[rollout_env, reward_env, train_env, part1_train_env],
    env_vars=_PROFILE_ENV_VARS,
)


async def _generate_and_score(
    base: flyte.io.Dir,
    pairs: list[tuple[str, str]],
    iteration: int,
    adapter: flyte.io.Dir,
    version: int,
    seed: int,
) -> tuple[list[Rollout], list[float]]:
    """Fan out one iteration's rollouts on the warm pool and score groups as they finish."""
    import random

    prompts = random.Random(seed * 100003 + iteration).sample(pairs, min(PROMPTS_PER_ITER, len(pairs)))
    futs = [
        asyncio.create_task(generate(base, q, a, adapter, version, group_id=gid))
        for gid, (q, a) in enumerate(prompts)
    ]
    flat: list[Rollout] = []
    reward_futs: list[asyncio.Task[list[float]]] = []
    for fut in asyncio.as_completed(futs):
        group = await fut
        flat.extend(group)
        reward_futs.append(asyncio.create_task(score_group(group)))
    group_rewards = await asyncio.gather(*reward_futs)
    return flat, [r for gr in group_rewards for r in gr]


async def _timed_generate_and_score(*args: Any) -> tuple[list[Rollout], list[float], float]:
    t0 = time.monotonic()
    rollouts, rewards = await _generate_and_score(*args)
    return rollouts, rewards, time.monotonic() - t0


async def _publish(history: list[IterationMetrics], status: str, num_iterations: int, extra: dict) -> None:
    await flyte.report.replace.aio(
        render_report(
            history,
            status=status,
            base_model=BASE_MODEL_REPO,
            num_iterations=num_iterations,
            group_size=GROUP_SIZE,
            prompts_per_iter=PROMPTS_PER_ITER,
            lora_rank=LORA_RANK,
            learning_rate=LEARNING_RATE,
            extra=extra,
        ),
        do_flush=True,
    )


# {{docs-fragment pipelined_driver}}
@driver_env.task(report=True)
async def train_rl_clustered(
    base: flyte.io.Dir,
    num_iterations: int = NUM_ITERATIONS,
    dataset: str = DEFAULT_DATASET,
    pipelined: bool = True,
    seed: int = 17,
) -> flyte.io.Dir:
    """The Part-2 loop: train_step_clustered(N) runs CONCURRENTLY with generation of batch N+1.

    Pipeline (one-step off-policy — batch N+1 is generated with the adapter from step N-1; the
    trainer's clipped importance ratio is what makes that sound):

        sequential:  [gen 0][train 0][gen 1][train 1]...
        pipelined:   [gen 0][gen 1  ][gen 2  ]...
                            [train 0][train 1]...   <- JobSet cold start hidden under generation
                                     [eval 0 ][eval 1 ]  <- eval is off the critical path in BOTH modes

    ``pipelined=False`` runs the identical loop strictly sequentially, so two runs give the
    timing comparison chart. The held-out eval of step N is launched as a background task and
    only awaited after step N+1's training, so it never sits between iterations; its row in the
    report is back-filled one iteration late. Loop state (adapter, optimizer state, history,
    baseline eval) is checkpointed each iteration so a preempted driver resumes mid-run.
    """
    ctx = flyte.ctx()
    cp = ctx.checkpoint if ctx is not None else None

    train_pairs, eval_pairs = _load_pairs(dataset)
    eval_qs = [q for q, _ in eval_pairs]
    eval_as = [a for _, a in eval_pairs]

    start_iter = 0
    version = 0
    adapter: flyte.io.Dir | None = None
    opt_state: flyte.io.Dir | None = None
    baseline_accuracy: float | None = None
    history: list[IterationMetrics] = []

    # Resume from a prior driver attempt, if any.
    if cp is not None:
        prev = await cp.load()
        if prev is not None:
            state = json.loads(prev.read_text())
            start_iter = state["iteration"] + 1
            version = state["adapter_version"]
            adapter = flyte.io.Dir.from_existing_remote(state["adapter_path"])
            if state.get("opt_state_path"):
                opt_state = flyte.io.Dir.from_existing_remote(state["opt_state_path"])
            baseline_accuracy = state.get("baseline_accuracy")
            history = [IterationMetrics(**row) for row in state.get("history", [])]
            logger.info("Resumed from checkpoint at iteration %d (adapter v%d)", start_iter, version)

    if adapter is None:
        adapter = await init_adapter(base)
        version = 0

    # Pre-training baseline on the held-out set (base model, no adapter). Checkpoint-guarded so a
    # resumed driver does not pay for it twice.
    if baseline_accuracy is None:
        with flyte.group("eval-baseline"):
            baseline = await evaluate(base, eval_qs, eval_as, adapter=None)
        baseline_accuracy = baseline.accuracy

    extra = dict(
        profile=PROFILE_NAME,
        dataset=dataset,
        pipelined=pipelined,
        replicas=REPLICAS,
        nproc_per_node=NPROC_PER_NODE,
        clip_eps=CLIP_EPS,
        kl_beta=KL_BETA,
        baseline_accuracy=baseline_accuracy,
    )
    await _publish(history, "running", num_iterations, extra)

    # The in-flight held-out eval (for the most recently trained adapter). On resume the previous
    # attempt's in-flight eval is lost, so re-launch it if the last row is still un-scored.
    pending_eval: asyncio.Task | None = None
    if history and history[-1].eval_accuracy is None:
        with flyte.group(f"iter-{history[-1].iteration}-eval"):
            pending_eval = asyncio.create_task(evaluate(base, eval_qs, eval_as, adapter=adapter, version=version))

    async def _settle_eval() -> None:
        """Await the in-flight eval and back-fill its row (the row for the previous train step)."""
        nonlocal pending_eval
        if pending_eval is None:
            return
        ev = await pending_eval
        pending_eval = None
        for row in reversed(history):
            if row.eval_accuracy is None:
                row.eval_accuracy = ev.accuracy
                logger.info("iter %d: held-out eval %.1f%%", row.iteration, 100 * ev.accuracy)
                break

    # Prime the pipeline: the first batch with the current adapter.
    pending: asyncio.Task | None = None
    last_state: dict | None = None
    if start_iter < num_iterations:
        with flyte.group(f"iter-{start_iter}-generate"):
            pending = asyncio.create_task(
                _timed_generate_and_score(base, train_pairs, start_iter, adapter, version, seed)
            )

    for it in range(start_iter, num_iterations):
        t_iter = time.monotonic()
        assert pending is not None
        rollouts, rewards, gen_s = await pending

        # PIPELINED: kick off the NEXT batch immediately with the CURRENT adapter (one step stale
        # once training below lands) — the overlap covers the clustered trainer's whole lifetime,
        # JobSet spin-up included.
        if pipelined and it + 1 < num_iterations:
            with flyte.group(f"iter-{it + 1}-generate"):
                pending = asyncio.create_task(
                    _timed_generate_and_score(base, train_pairs, it + 1, adapter, version, seed)
                )

        # Train on batch N (while batch N+1 generates on the warm pool, if pipelined).
        t_train = time.monotonic()
        version += 1
        with flyte.group(f"iter-{it}-train"):
            adapter, opt_state, tm = await train_step_clustered(
                base, rollouts, rewards, adapter, version, opt_state
            )
        train_s = time.monotonic() - t_train
        iter_s = time.monotonic() - t_iter  # gen (residual) + train; eval is off the critical path

        # SEQUENTIAL: only now start the next batch, with the freshly-trained adapter.
        if not pipelined and it + 1 < num_iterations:
            with flyte.group(f"iter-{it + 1}-generate"):
                pending = asyncio.create_task(
                    _timed_generate_and_score(base, train_pairs, it + 1, adapter, version, seed)
                )

        # The PREVIOUS step's eval overlapped this step's training; settle it now (usually done).
        # Then launch this step's eval in the background — it overlaps the NEXT train step.
        await _settle_eval()
        with flyte.group(f"iter-{it}-eval"):
            pending_eval = asyncio.create_task(
                evaluate(base, eval_qs, eval_as, adapter=adapter, version=version)
            )

        n = len(rollouts)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        correct = sum(1 for r in rollouts if _extract_answer(r.completion) == r.answer)
        formatted = sum(1 for r in rollouts if "####" in r.completion)
        best_idx = max(range(n), key=lambda i: rewards[i]) if n else None
        history.append(
            IterationMetrics(
                iteration=it,
                adapter_version=version,
                num_rollouts=n,
                mean_reward=mean_reward,
                max_reward=max(rewards) if rewards else 0.0,
                accuracy=correct / n if n else 0.0,
                format_rate=formatted / n if n else 0.0,
                mean_loss=tm.mean_loss,
                contributing=tm.contributing,
                sample_question=rollouts[best_idx].question if best_idx is not None else "",
                sample_completion=rollouts[best_idx].completion if best_idx is not None else "",
                sample_reward=rewards[best_idx] if best_idx is not None else 0.0,
                eval_accuracy=None,  # back-filled by _settle_eval after the next train step
                mean_ratio=tm.mean_ratio,
                clip_fraction=tm.clip_fraction,
                mean_kl=tm.mean_kl,
                gen_seconds=gen_s,
                train_seconds=train_s,
                iter_seconds=iter_s,
            )
        )
        logger.info(
            "iter %d: reward %.3f, loss %.4f, ratio %.3f, clip %.3f, KL %.4f, gen %.0fs, train %.0fs, iter %.0fs",
            it, mean_reward, tm.mean_loss, tm.mean_ratio, tm.clip_fraction, tm.mean_kl, gen_s, train_s, iter_s,
        )
        await _publish(history, "running", num_iterations, extra)

        # Persist loop state so a preempted driver resumes here (with its report history).
        if cp is not None:
            last_state = {
                "iteration": it,
                "adapter_version": version,
                "adapter_path": adapter.path,
                "opt_state_path": opt_state.path if opt_state is not None else None,
                "baseline_accuracy": baseline_accuracy,
                "dataset": dataset,
                "seed": seed,
                "history": [vars(m) for m in history],
            }
            await cp.save(json.dumps(last_state).encode())

    # The final step's eval has nothing left to hide under — wait for it so the report is complete.
    await _settle_eval()
    if cp is not None and last_state is not None:
        last_state["history"] = [vars(m) for m in history]
        await cp.save(json.dumps(last_state).encode())
    await _publish(history, "complete", num_iterations, extra)
    assert adapter is not None
    return adapter
# {{/docs-fragment pipelined_driver}}


if __name__ == "__main__":
    import argparse
    import inspect

    import flyte.prefetch

    parser = argparse.ArgumentParser(
        description=f"Part 2: async multi-node GRPO (profile={PROFILE_NAME}; set GRPO_PROFILE=stress to scale)"
    )
    parser.add_argument("--dataset", choices=["arithmetic", "gsm8k"], default=DEFAULT_DATASET)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--sequential", action="store_true", help="disable pipelining (for the timing comparison)")
    args = parser.parse_args()

    flyte.init_from_config()
    run = flyte.prefetch.hf_model(repo=BASE_MODEL_REPO, hf_token_key="hf-token")
    run.wait()
    outputs = run.outputs()
    if inspect.isawaitable(outputs):
        outputs = asyncio.run(outputs)

    rl_run = flyte.run(
        train_rl_clustered,
        base=outputs[0],
        num_iterations=args.iterations,
        dataset=args.dataset,
        pipelined=not args.sequential,
    )
    print(rl_run.url)
    rl_run.wait()
