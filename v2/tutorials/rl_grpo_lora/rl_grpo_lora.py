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
# ]
# main = "train_rl"
# params = ""
# ///
"""
GRPO + LoRA reinforcement learning for LLMs on Union with flyte-sdk
==================================================================

A runnable reinforcement-learning loop for LLMs (GRPO with LoRA), orchestrated by ordinary Flyte async
tasks. The sibling ``README.md`` is a step-by-step tutorial that explains the concepts; this module is
the implementation. The loop has the standard RL-for-LLMs shape::

    sample prompts -> generate rollouts (vLLM) -> score (reward) -> policy update (LoRA) -> refresh adapter -> repeat

Flyte with Union fits this well: each stage runs in its own right-sized environment (GPU rollouts and
trainer, CPU reward and driver), the driver is just an ``async`` ``for`` loop, the run is resumable via
``flyte.Checkpoint``, and progress streams to a live ``flyte.report``. The one technique unique to
RL-for-LLMs is *keeping vLLM warm in a reusable container and swapping the LoRA adapter each
iteration* — everything else is ordinary Flyte.

What runs where
---------------
- ``generate``   — **reusable, warm** ``TaskEnvironment`` (``ReusePolicy``). Holds an in-process vLLM
                   ``LLM(enable_lora=True)`` engine as a module global so the frozen base never
                   reloads; the per-iteration LoRA adapter is attached per request via ``LoRARequest``.
- ``score_group``— plain CPU ``@task``, rule-based / verifiable reward (exact-match + format). Scores a
                   whole prompt group per call (one reward task per group, not per rollout).
- ``init_adapter`` / ``train_step`` — single-node GPU ``@task``; one GRPO step over the
                   externally-generated rollouts, training only the PEFT LoRA adapter (base frozen),
                   returning the new adapter as a ``flyte.io.Dir`` (a few MB).
- ``train_rl``   — the driver: a plain async ``@task`` running ``for it in range(N)``, fanning out
                   rollouts with ``asyncio.create_task`` and scoring each the moment it finishes with
                   ``asyncio.as_completed`` (no ``flyte.map`` barrier), then calling ``train_step``.
                   Resumes loop state from ``flyte.Checkpoint`` and wraps each iteration in
                   ``flyte.group(f"iter-{it}")``.

Two implementation choices worth knowing (also explained in the README):

1. **A custom GRPO step rather than a library trainer (e.g. TRL ``GRPOTrainer``).** Library trainers
   own the whole loop — they generate completions *and* call the reward function internally — so they
   can't consume the rollouts+rewards we produce in separate tasks, and would bypass the warm vLLM
   pool and the as-completed pipelining that are the point of running this on Flyte. The loss here is
   the standard group-normalized policy gradient (advantage = ``(r - mean_group)/(std_group + eps)``),
   trained through the PEFT LoRA params only. See ``train_step``.

2. **Plain-HF prefetch rather than vLLM-sharded weights.** vLLM's pre-sharded layout
   (``model-rank-*-part-*.safetensors``) is *not* readable by the HF/PEFT trainer, and for
   ``tensor_parallel_size=1`` + a small model vLLM loads plain HF weights directly with no benefit
   lost. Pre-sharding only pays off for TP>1 rollout replicas, which would then need a *separate* HF
   copy of the base for the trainer. Here one plain-HF ``Dir`` feeds both generator and trainer.

API verification
----------------
Signatures below were checked against source, not assumed:
- ``flyte`` (src/flyte): ``TaskEnvironment``, ``ReusePolicy(replicas, idle_ttl, concurrency,
  scaledown_ttl)``, ``Resources``, ``GPU``, ``Secret(key, as_env_var)``, ``Image.from_debian_base``,
  ``Checkpoint`` (``await load()/save(bytes|path)``), ``group``, ``io.Dir`` (``download``,
  ``from_local``, ``from_existing_remote``), ``prefetch.hf_model`` (returns a ``Run``; the output
  ``Dir`` is ``(await run.outputs())[0]``).
- vLLM 0.11.0 (uv cache): ``vllm.LLM(model=..., enable_lora=True, max_lora_rank=...)``,
  ``LLM.generate(prompts, sampling_params, lora_request=...)``,
  ``vllm.lora.request.LoRARequest(lora_name, lora_int_id, lora_path)`` (``lora_int_id >= 1``).
- PEFT (stable API): ``LoraConfig`` / ``get_peft_model`` / ``PeftModel.from_pretrained(...,
  is_trainable=True)`` / ``save_pretrained``.

Run::

    # remote (needs a GPU-backed Union deployment + an HF token secret named `hf-token`)
    python rl_grpo_lora.py

Validated end to end on a Union demo cluster (Qwen3-0.6B, L4 GPUs, 3 GRPO iterations): prefetch →
init_adapter → 12 warm-vLLM rollouts → 72 pipelined reward tasks → 3 GRPO train steps → final LoRA
adapter (v3) returned as a flyte.Dir, with the live flyte.report published each iteration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Any

from async_lru import alru_cache
from pydantic import BaseModel

import flyte
import flyte.io
import flyte.report
from report_helpers import IterationMetrics, render_report

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------
# Configuration (small + cheap on purpose — this is an MVP meant to validate the wiring end to end)
# ----------------------------------------------------------------------------------------------------
BASE_MODEL_REPO = "Qwen/Qwen3-0.6B"  # small, fast; swap for a bigger policy once the loop is proven

NUM_ITERATIONS = 3  # GRPO outer steps the driver runs
PROMPTS_PER_ITER = 4  # prompts (= GRPO groups) sampled per iteration
GROUP_SIZE = 6  # completions per prompt; GRPO normalizes advantage within each group
MAX_NEW_TOKENS = 256
SAMPLING_TEMPERATURE = 1.0  # >0 so the group has diverse samples to rank

LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-5

# Fixed prompt template shared verbatim by the rollout (vLLM) and trainer (HF) sides so completion
# tokens line up. Kept deliberately tokenizer-agnostic (no chat template) to avoid vLLM/HF drift.
SYSTEM_PREAMBLE = (
    "You are a careful math assistant. Reason briefly, then give the final integer answer on its own "
    "line prefixed by '#### '."
)


def build_prompt(question: str) -> str:
    """Render a question into the exact text both rollout and trainer condition on."""
    return f"{SYSTEM_PREAMBLE}\n\nQuestion: {question}\nAnswer:"


# A tiny, fully verifiable dataset (no external download → deterministic + cheap). Each entry is
# (question, ground-truth integer answer as a string).
DATASET: list[tuple[str, str]] = [
    ("What is 12 + 7?", "19"),
    ("What is 9 * 6?", "54"),
    ("What is 45 - 18?", "27"),
    ("What is 100 / 4?", "25"),
    ("What is 13 + 28?", "41"),
    ("What is 7 * 8?", "56"),
    ("What is 81 - 36?", "45"),
    ("What is 144 / 12?", "12"),
]


# ----------------------------------------------------------------------------------------------------
# Rollout payload — one sampled completion for one prompt, tagged with its GRPO group.
# ----------------------------------------------------------------------------------------------------
class Rollout(BaseModel):
    group_id: int  # which prompt group this completion belongs to (advantage is normalized per group)
    question: str  # raw question (reward + trainer re-render the prompt via build_prompt)
    completion: str  # text the policy generated
    answer: str  # ground-truth answer, for the verifiable reward


# ----------------------------------------------------------------------------------------------------
# Images & environments
# ----------------------------------------------------------------------------------------------------
# One image shared by every env. Built explicitly (rather than from the uv-script header) so we can
# pull vLLM's flashinfer kernels as *precompiled cubin wheels* — without them vLLM tries to JIT-compile
# attention at runtime and fails with "Could not find nvcc" (no CUDA toolkit in the base image). This
# recipe mirrors the proven examples/genai/vllm/vllm_app.py. torch comes transitively from vllm.
# `unionai-reuse` provides the actor bridge required by the reusable rollout env. The module top level
# only imports flyte + pydantic; torch/vllm/transformers/peft are imported lazily inside the GPU tasks.
image = (
    flyte.Image.from_debian_base(name="rl-grpo-lora")
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
    .with_pip_packages(
        "vllm==0.11.0",
        "transformers==4.57.6",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "unionai-reuse>=0.1.3",
        "async-lru>=2.0.0",
    )
)

HF_SECRET = flyte.Secret(key="hf-token", as_env_var="HF_TOKEN")

# Rollout generator: warm, reusable vLLM. concurrency=1 because a single in-process vLLM engine
# batches internally and is not safe to drive from several coroutines at once; the driver still
# pipelines by fanning generate() calls across replicas.
rollout_env = flyte.TaskEnvironment(
    name="rl-grpo-rollout",
    image=image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=flyte.GPU("L4", 1), shm="auto"),
    reusable=flyte.ReusePolicy(replicas=(1, 4), concurrency=1, idle_ttl=300, scaledown_ttl=120),
    secrets=[HF_SECRET],
    env_vars={"VLLM_USE_V1": "1"},
)

# Reward: cheap, rule-based, CPU only.
reward_env = flyte.TaskEnvironment(
    name="rl-grpo-reward",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

# Trainer: single node, one GPU is plenty for a 0.6B base + LoRA.
train_env = flyte.TaskEnvironment(
    name="rl-grpo-train",
    image=image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=flyte.GPU("L4", 1), shm="auto"),
    secrets=[HF_SECRET],
)

# Driver: plain async orchestration, no GPU. It invokes tasks in the rollout/reward/train envs, so it
# must declare them via depends_on so their images/environments are registered alongside the driver's.
driver_env = flyte.TaskEnvironment(
    name="rl-grpo-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[rollout_env, reward_env, train_env],
)

# ----------------------------------------------------------------------------------------------------
# 1. Rollout generation — warm in-process vLLM with per-request LoRA
# ----------------------------------------------------------------------------------------------------
# The expensive per-replica work is cached with @alru_cache, so it happens once and is reused across
# every generate() call a warm replica handles. We key the caches on the remote URI string (hashable)
# rather than the flyte.io.Dir object. Returns are typed Any because vLLM is imported lazily.


@alru_cache(maxsize=1)
async def _load_engine(base_uri: str) -> Any:
    """Build the vLLM engine once per warm replica (cached); the frozen base stays resident in GPU."""
    from vllm import LLM

    local_base: str = await flyte.io.Dir.from_existing_remote(base_uri).download()  # plain-HF base
    logger.info("Building warm vLLM engine from %s", local_base)
    return LLM(
        model=local_base,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=True,  # skip CUDA-graph capture → faster cold start for an MVP
    )


@alru_cache(maxsize=None)
async def _adapter_local_path(adapter_uri: str) -> str:
    """Download a LoRA adapter once per warm replica (cached by its remote URI → one download/version)."""
    return await flyte.io.Dir.from_existing_remote(adapter_uri).download()


@rollout_env.task
async def generate(
    base: flyte.io.Dir,
    question: str,
    answer: str,
    adapter: flyte.io.Dir,
    version: int,
    group_id: int,
) -> list[Rollout]:
    """Generate a GROUP_SIZE group of completions for one prompt, using the current LoRA adapter.

    The frozen base loads exactly once per replica (cached ``_load_engine``); each adapter version is
    downloaded once (cached ``_adapter_local_path``) and attached per request via ``LoRARequest`` — the
    base weights in GPU memory are never touched.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    engine: Any = await _load_engine(base.path)
    adapter_path: str = await _adapter_local_path(adapter.path)

    # lora_int_id must be >= 1 and unique per adapter; version starts at 0 so shift by 1.
    lora: LoRARequest = LoRARequest(f"policy-v{version}", version + 1, adapter_path)
    sampling: SamplingParams = SamplingParams(
        n=GROUP_SIZE,
        temperature=SAMPLING_TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    prompt: str = build_prompt(question)
    # vLLM's generate() is a blocking call, so we run it via asyncio.to_thread to keep it off the event
    # loop — that lets the reusable replica's background actor heartbeat stay responsive while the GPU
    # is busy. (We deliberately do not use flyte.extras.DynamicBatcher here: it batches many concurrent
    # producers, whereas this env runs concurrency=1 and each call already submits a full group of
    # GROUP_SIZE sequences as one vLLM batch.)
    outputs: Any = await asyncio.to_thread(engine.generate, [prompt], sampling, lora_request=lora)
    completions: list[str] = [o.text for o in outputs[0].outputs]
    logger.info("group %s: generated %d completions (adapter v%d)", group_id, len(completions), version)
    return [
        Rollout(group_id=group_id, question=question, completion=c, answer=answer) for c in completions
    ]


# ----------------------------------------------------------------------------------------------------
# 2. Reward — rule-based / verifiable
# ----------------------------------------------------------------------------------------------------
def _extract_answer(text: str) -> str | None:
    """Pull the integer following the last '####' marker; fall back to the last integer in the text."""
    import re

    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = re.search(r"-?\d+", tail)
        if m:
            return m.group(0)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def _reward(rollout: Rollout) -> float:
    """Verifiable reward: 1.0 for the correct answer, +0.2 format bonus for emitting the '####' marker."""
    reward = 0.0
    if "####" in rollout.completion:
        reward += 0.2
    predicted = _extract_answer(rollout.completion)
    if predicted is not None and predicted == rollout.answer:
        reward += 1.0
    return reward


@reward_env.task
async def score_group(rollouts: list[Rollout]) -> list[float]:
    """Score a whole prompt group in one task — one reward task per group, not per rollout.

    The rule-based reward is microseconds of pure-Python work, so a task *per rollout* would pay pod
    startup over and over for trivial compute. Scoring at the group granularity (the unit `generate`
    already returns) keeps reward an observable, pipelined task while cutting the pod count ~GROUP_SIZE×.
    """
    return [_reward(r) for r in rollouts]


# ----------------------------------------------------------------------------------------------------
# 3. Trainer — one GRPO step on the PEFT LoRA adapter (base frozen)
# ----------------------------------------------------------------------------------------------------
@train_env.task
async def init_adapter(base: flyte.io.Dir) -> flyte.io.Dir:
    """Create a fresh (untrained) LoRA adapter so iteration 0 already has an adapter to attach."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    local_base: str = await base.download()
    model: Any = AutoModelForCausalLM.from_pretrained(
        local_base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    lora_config: Any = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    out_dir: str = tempfile.mkdtemp(prefix="adapter-v0-")
    model.save_pretrained(out_dir)  # writes adapter_config.json + adapter_model.safetensors only
    logger.info("Initialized fresh LoRA adapter at %s", out_dir)
    return await flyte.io.Dir.from_local(out_dir)


def _group_normalized_advantages(rollouts: list[Rollout], rewards: list[float]) -> list[float]:
    """GRPO advantage: within each prompt group, ``(r - mean) / (std + eps)``."""
    import statistics
    from collections import defaultdict

    by_group: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rollouts):
        by_group[r.group_id].append(i)

    advantages = [0.0] * len(rollouts)
    for idxs in by_group.values():
        group_rewards = [rewards[i] for i in idxs]
        mean = statistics.fmean(group_rewards)
        std = statistics.pstdev(group_rewards) if len(group_rewards) > 1 else 0.0
        for i in idxs:
            advantages[i] = (rewards[i] - mean) / (std + 1e-4)
    return advantages


@train_env.task
async def train_step(
    base: flyte.io.Dir,
    rollouts: list[Rollout],
    rewards: list[float],
    adapter: flyte.io.Dir,
    version: int,
) -> tuple[flyte.io.Dir, float, int]:
    """One GRPO policy-gradient step over externally-generated rollouts; trains the LoRA adapter only.

    Resumes from the previous adapter (``PeftModel.from_pretrained(..., is_trainable=True)``), takes a
    single optimizer step on the group-normalized policy-gradient loss, and ``save_pretrained()``s the
    new adapter as a ``flyte.io.Dir``. See module docstring for why this is hand-rolled rather than TRL.

    Returns ``(new_adapter, mean_loss, contributing)`` so the driver can chart loss in the report.
    """
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_base: str = await base.download()
    local_adapter: str = await adapter.download()

    tokenizer: Any = AutoTokenizer.from_pretrained(local_base, trust_remote_code=True)
    base_model: Any = AutoModelForCausalLM.from_pretrained(
        local_base, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    # Resume the trainable adapter from the previous version (frozen base, only A/B train).
    model: Any = PeftModel.from_pretrained(base_model, local_adapter, is_trainable=True).to(device)
    model.train()

    advantages: list[float] = _group_normalized_advantages(rollouts, rewards)
    optimizer: Any = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE)
    optimizer.zero_grad()

    total_loss: float = 0.0
    contributing: int = 0
    for rollout, advantage in zip(rollouts, advantages):
        if advantage == 0.0:
            continue  # no learning signal (whole group scored identically)

        prompt_text = build_prompt(rollout.question)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        full_ids = tokenizer(prompt_text + rollout.completion, return_tensors="pt").input_ids.to(device)

        prompt_len = prompt_ids.shape[1]
        if full_ids.shape[1] <= prompt_len:
            continue  # empty completion after tokenization

        logits = model(full_ids).logits  # (1, seq, vocab)
        # log p(token_t | token_<t): align logits[:-1] with targets full_ids[1:]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = full_ids[:, 1:]
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq-1)

        # Mask to the completion tokens only (targets at positions >= prompt_len-1 in the shifted view).
        completion_mask = torch.zeros_like(token_log_probs)
        completion_mask[:, prompt_len - 1 :] = 1.0
        seq_log_prob = (token_log_probs * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        loss = -advantage * seq_log_prob
        loss.backward()  # accumulate gradients across the batch, single optimizer step below
        total_loss += float(loss.item())
        contributing += 1

    if contributing > 0:
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        logger.info(
            "GRPO step v%d: %d/%d rollouts contributed, mean loss %.4f",
            version,
            contributing,
            len(rollouts),
            total_loss / contributing,
        )
    else:
        logger.info("GRPO step v%d: no contributing rollouts (flat rewards); adapter unchanged", version)

    out_dir = tempfile.mkdtemp(prefix=f"adapter-v{version}-")
    model.save_pretrained(out_dir)
    mean_loss = total_loss / contributing if contributing > 0 else 0.0
    new_adapter = await flyte.io.Dir.from_local(out_dir)
    return new_adapter, mean_loss, contributing


# ----------------------------------------------------------------------------------------------------
# 4. Driver — the RL loop (replaces Ray)
# ----------------------------------------------------------------------------------------------------
def _sample_prompts(iteration: int) -> list[tuple[str, str]]:
    """Deterministically rotate through the dataset so each iteration sees a different slice."""
    start = (iteration * PROMPTS_PER_ITER) % len(DATASET)
    return [DATASET[(start + i) % len(DATASET)] for i in range(PROMPTS_PER_ITER)]


def _report_config() -> dict[str, Any]:
    """Static run config surfaced in the report header."""
    return dict(
        base_model=BASE_MODEL_REPO,
        num_iterations=NUM_ITERATIONS,
        group_size=GROUP_SIZE,
        prompts_per_iter=PROMPTS_PER_ITER,
        lora_rank=LORA_RANK,
        learning_rate=LEARNING_RATE,
    )


async def _publish_report(history: list[IterationMetrics], status: str) -> None:
    """Re-render and flush the live GRPO progress report to the driver task's report tab."""
    await flyte.report.replace.aio(
        render_report(history, status=status, **_report_config()),
        do_flush=True,
    )


@driver_env.task(report=True)
async def train_rl(base: flyte.io.Dir, num_iterations: int = NUM_ITERATIONS) -> flyte.io.Dir:
    """Own the GRPO loop: fan out rollouts, score as they finish, take one GRPO step, repeat.

    Loop state (iteration, current adapter, and the report history) is checkpointed each iteration so a
    preempted driver resumes mid-run — including the accumulated report rows — instead of restarting.
    Progress is published to a live HTML report (``report=True``) after every iteration.
    """
    ctx = flyte.ctx()
    cp = ctx.checkpoint if ctx is not None else None

    start_iter = 0
    adapter: flyte.io.Dir | None = None
    adapter_version = 0
    history: list[IterationMetrics] = []

    # Resume from a prior driver attempt, if any.
    if cp is not None:
        prev = await cp.load()
        if prev is not None:
            state = json.loads(prev.read_text())
            start_iter = state["iteration"] + 1
            adapter_version = state["adapter_version"]
            adapter = flyte.io.Dir.from_existing_remote(state["adapter_path"])
            history = [IterationMetrics(**row) for row in state.get("history", [])]
            logger.info("Resumed from checkpoint at iteration %d (adapter v%d)", start_iter, adapter_version)

    # Cold start: mint a fresh LoRA adapter (version 0).
    if adapter is None:
        adapter = await init_adapter(base)
        adapter_version = 0

    await _publish_report(history, status="running")

    for it in range(start_iter, num_iterations):
        with flyte.group(f"iter-{it}"):
            prompts = _sample_prompts(it)

            # Launch every rollout group at once on the warm replicas.
            rollout_futs = [
                asyncio.create_task(generate(base, q, a, adapter, adapter_version, group_id=gid))
                for gid, (q, a) in enumerate(prompts)
            ]

            # Score each group the instant its rollout finishes — reward overlaps in-flight rollouts.
            # One reward task per group (not per rollout): see score_group.
            flat_rollouts: list[Rollout] = []
            reward_futs: list[asyncio.Task[list[float]]] = []
            for fut in asyncio.as_completed(rollout_futs):
                group = await fut
                flat_rollouts.extend(group)
                reward_futs.append(asyncio.create_task(score_group(group)))

            group_rewards = await asyncio.gather(*reward_futs)  # aligned with append order
            rewards = [r for gr in group_rewards for r in gr]  # flatten → aligned with flat_rollouts
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            logger.info("iter %d: %d rollouts, mean reward %.3f", it, len(rewards), mean_reward)

            # One GRPO step → next adapter version.
            new_version = adapter_version + 1
            adapter, mean_loss, contributing = await train_step(
                base, flat_rollouts, rewards, adapter, new_version
            )
            adapter_version = new_version

            # Record metrics for the report (accuracy/format derived directly from the rollouts).
            n = len(flat_rollouts)
            correct = sum(1 for r in flat_rollouts if _extract_answer(r.completion) == r.answer)
            formatted = sum(1 for r in flat_rollouts if "####" in r.completion)
            best_idx = max(range(n), key=lambda i: rewards[i]) if n else None
            history.append(
                IterationMetrics(
                    iteration=it,
                    adapter_version=adapter_version,
                    num_rollouts=n,
                    mean_reward=mean_reward,
                    max_reward=max(rewards) if rewards else 0.0,
                    accuracy=correct / n if n else 0.0,
                    format_rate=formatted / n if n else 0.0,
                    mean_loss=mean_loss,
                    contributing=contributing,
                    sample_question=flat_rollouts[best_idx].question if best_idx is not None else "",
                    sample_completion=flat_rollouts[best_idx].completion if best_idx is not None else "",
                    sample_reward=rewards[best_idx] if best_idx is not None else 0.0,
                )
            )
            await _publish_report(history, status="running")

            # Persist loop state so a preempted driver resumes here (with its report history).
            if cp is not None:
                state = {
                    "iteration": it,
                    "adapter_version": adapter_version,
                    "adapter_path": adapter.path,
                    "history": [vars(m) for m in history],
                }
                await cp.save(json.dumps(state).encode())

    await _publish_report(history, status="complete")
    assert adapter is not None
    return adapter


# ----------------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import flyte.prefetch

    flyte.init_from_config()

    # Prefetch the base ONCE into the Flyte object store as plain HF weights (see module docstring for
    # why we do not vLLM-shard for this single-GPU MVP). hf_model returns a Run; its sole output is the
    # model Dir, which we pass straight into the driver task as a flyte.io.Dir.
    run = flyte.prefetch.hf_model(repo=BASE_MODEL_REPO, hf_token_key="hf-token")
    run.wait()
    print(f"Prefetched base model: {run.url}")
    # hf_model's sole output is the model Dir. run.outputs() may be sync or awaitable depending on the
    # SDK build, so handle both. The result is an ActionOutputs tuple; element 0 is the base Dir.
    import inspect

    outputs = run.outputs()
    if inspect.isawaitable(outputs):
        outputs = asyncio.run(outputs)
    base_dir = outputs[0]

    rl_run = flyte.run(train_rl, base=base_dir, num_iterations=NUM_ITERATIONS)
    print(rl_run.url)
    rl_run.wait()
