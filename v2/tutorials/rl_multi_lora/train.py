# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.5.0",
#    "flyteplugins-vllm",
#    "unionai-reuse>=0.1.3",
#    "async-lru>=2.0.0",
# ]
# ///
"""
Train several LoRA adapters with reinforcement learning on one shared set of base-model weights.

    python train.py --task all            # one run per skill, sharing the same warm GPU replicas
    python train.py --task countdown      # or --task json

This is the training half of the tutorial in README.md. Read that first if the terms below are new.

What happens in a run
---------------------
A small base model (Qwen2.5-1.5B-Instruct) is bad at a skill. The run measures how bad, improves it with
reinforcement learning, measures again, and publishes the result as a model you can serve:

    eval base -> [ sample problems -> generate answers -> score them -> one GRPO step ] x N -> eval
              -> a person reviews the result -> publish

Only a LoRA adapter is trained: a few megabytes of extra weights layered on the frozen base. The reward
is a plain Python function that *checks* an answer (``tasks.score``), so no labeled solutions are needed.
That is "RL with verifiable rewards" (RLVR). To use human preferences or a judge model instead (RLHF),
you replace that one function; the rest of the loop is unchanged. README.md shows how.

What Flyte and Union do for you
-------------------------------
- **A task environment per kind of work.** Rollouts and training get a GPU; reward and the driver get
  a small CPU container. Each is declared in a few lines under "Images & environments".
- **Warm, reusable containers.** Loading a model takes far longer than using it once. The rollout and
  trainer environments are *reusable*: the container stays alive between calls with the model already
  in GPU memory, so iteration 2 onwards never loads anything.
- **One set of base weights for every skill.** Reusable environments are shared across runs, so the
  countdown run and the JSON run use the same replicas. One vLLM engine serves both skills' rollouts and
  evals (the LoRA is attached per request). One trainer holds the frozen base once, with a LoRA adapter
  and optimizer state per skill.
- **Resumable loops.** The driver checkpoints after every iteration; a preempted run picks up where
  it stopped.
- **A live report and a versioned model.** The driver streams reward curves and before/after accuracy
  to the run page, then publishes the merged model as an artifact that ``serve.py`` deploys by name.
- **A review per run.** Before publishing, each run pauses on a condition (``flyte.new_condition``)
  until a person approves or rejects that skill's model. Nothing is published without an approval.

Details worth knowing before you change the code
------------------------------------------------
- **Chat-template prompts.** Rollouts and training render prompts with the model's own chat template,
  so the trained model behaves the same behind an OpenAI-style chat endpoint as it did during RL.
- **Exact sampled tokens.** The trainer scores the token ids vLLM actually sampled instead of
  re-tokenizing the text, so the gradient is for the tokens the policy really chose.
- **Batched rollouts.** Each ``generate`` call submits a whole chunk of prompts as one vLLM batch.

Forked from the single-skill ``rl_grpo_lora`` tutorial (unionai-examples/v2/tutorials/rl_grpo_lora).
The ``docs-fragment`` comments mark the snippets the documentation page embeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import flyte
import flyte.io
import flyte.report
from async_lru import alru_cache
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE
from pydantic import BaseModel

import tasks
from report_helpers import EvalResult, IterationMetrics, render_report

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment config}}
BASE_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
# The name flyte.prefetch.hf_model publishes for BASE_MODEL_REPO (repo tail, '.' -> '-').
BASE_ARTIFACT = "Qwen2-5-1-5B-Instruct"
# The model artifact each skill's run publishes; serve.py deploys them by these names.
MODEL_ARTIFACTS = {skill: f"qwen2-5-1-5b-rl-{skill}" for skill in tasks.TASKS}


@dataclass(frozen=True)
class TaskConfig:
    iterations: int
    max_new_tokens: int


TASK_CONFIG = {
    "countdown": TaskConfig(iterations=60, max_new_tokens=768),  # needs room to think; truncated = 0 reward
    "json": TaskConfig(iterations=30, max_new_tokens=320),  # one JSON object; converges fast
}

# One iteration = PROMPTS_PER_ITER problems x GROUP_SIZE sampled answers each = 128 rollouts, then one
# gradient step.
PROMPTS_PER_ITER = 16  # prompts (= GRPO groups) per iteration
GROUP_SIZE = 8  # completions per prompt; advantage is normalized within each group
ROLLOUT_CHUNKS = 2  # generate() calls per iteration, spread across rollout replicas
SAMPLING_TEMPERATURE = 1.0
EVAL_SIZE = 100

# LoRA: train small low-rank matrices next to these layers and leave the base weights frozen.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 3e-5
MICRO_BATCH = 2  # sequences per forward/backward; full-vocab logits dominate memory on an L4

# How long a finished run waits for a reviewer before it gives up without publishing.
REVIEW_TIMEOUT = timedelta(hours=24)
# {{/docs-fragment config}}


# {{docs-fragment rollout_type}}
class Rollout(BaseModel):
    group_id: int  # which prompt group this completion belongs to (advantage is normalized per group)
    question: str  # user message; the trainer re-renders the prompt from it
    answer: str  # ground truth for the verifiable reward (JSON, see tasks.py)
    completion: str
    completion_ids: list[int]  # exactly what vLLM sampled, including the stop token


# {{/docs-fragment rollout_type}}


# ----------------------------------------------------------------------------------------------------
# Images & environments
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment image}}
# Built on the vLLM plugin's serving image, so rollouts here and serving in serve.py run the same vLLM
# (and share its layers in the build cache). It already carries vLLM's own torch/transformers pins and
# the prebuilt flashinfer kernels; training only adds peft and the reusable-container bridge.
image = DEFAULT_VLLM_IMAGE.clone(name="rl-demo-train").with_pip_packages(
    "peft>=0.21",
    "accelerate>=1.0",
    "unionai-reuse>=0.1.3",
    "async-lru>=2.0.0",
)

# 24Gi, not 32Gi: a g6.2xlarge (1x L4) has ~29Gi allocatable, and anything larger spills over to the
# 4x L4 g6.12xlarge, which is often out of capacity.
L4 = flyte.Resources(cpu=4, memory="24Gi", gpu=flyte.GPU("L4", 1), shm="auto")
# {{/docs-fragment image}}


# {{docs-fragment rollout_env}}
# The reusable environments below keep ReusePolicy's default scope="global": every run of this file lands
# on the same warm replicas, so concurrent countdown and JSON runs share one copy of the base weights.
# Nothing in a replica may therefore assume it serves a single run; state is keyed by skill / adapter path.

# Rollouts and eval: warm vLLM with per-request LoRA. concurrency=1 because one in-process engine is not
# safe to drive from several coroutines; the driver parallelizes across replicas instead.
rollout_env = flyte.TaskEnvironment(
    name="rl-demo-rollout",
    image=image,
    resources=L4,
    reusable=flyte.ReusePolicy(replicas=(1, ROLLOUT_CHUNKS), concurrency=1, idle_ttl=600, scaledown_ttl=300),
)
# {{/docs-fragment rollout_env}}


# {{docs-fragment reward_env}}
# Reward: microseconds of pure Python, so one warm CPU container serves every scoring call.
reward_env = flyte.TaskEnvironment(
    name="rl-demo-reward",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=8, idle_ttl=600),
)
# {{/docs-fragment reward_env}}


# {{docs-fragment train_env}}
# Trainer: one warm GPU replica that keeps the base model resident, plus a LoRA adapter and optimizer per
# skill. concurrency=2 lets one step per skill be in flight: the GPU part of a step is serialized by
# _TRAINER_LOCK (the active adapter is a switch on the shared model, and two backward passes don't fit
# on an L4), while input download and adapter upload overlap with the other skill's GPU work.
train_env = flyte.TaskEnvironment(
    name="rl-demo-train",
    image=image,
    resources=L4,
    reusable=flyte.ReusePolicy(replicas=1, concurrency=len(tasks.TASKS), idle_ttl=600, scaledown_ttl=300),
)
# {{/docs-fragment train_env}}


# {{docs-fragment driver_env}}
driver_env = flyte.TaskEnvironment(
    name="rl-demo-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[rollout_env, reward_env, train_env],
)
# {{/docs-fragment driver_env}}


# ----------------------------------------------------------------------------------------------------
# Shared helpers (GPU-side imports stay lazy so the driver and reward containers stay light)
# ----------------------------------------------------------------------------------------------------
@alru_cache(maxsize=4)
async def _local_dir(uri: str) -> str:
    """Download a remote Dir once per warm replica (base weights, adapter versions)."""
    return await flyte.io.Dir.from_existing_remote(uri).download()


def _render(tokenizer: Any, task: str, question: str) -> str:
    """The exact prompt text both vLLM and the trainer condition on."""
    return tokenizer.apply_chat_template(tasks.messages(task, question), tokenize=False, add_generation_prompt=True)


# ----------------------------------------------------------------------------------------------------
# 1. Rollouts + eval — warm in-process vLLM, LoRA attached per request
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment engine_cache}}
@alru_cache(maxsize=1)
async def _load_engine(base_uri: str) -> Any:
    """Build the vLLM engine once per warm replica; the frozen base stays resident on the GPU."""
    from vllm import LLM

    local_base = await _local_dir(base_uri)
    logger.info("Building warm vLLM engine from %s", local_base)
    return LLM(
        model=local_base,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enable_prefix_caching=True,  # every prompt in a task shares the long system prompt
    )


# {{/docs-fragment engine_cache}}


async def _generate(engine: Any, prompts: list[str], sampling: Any, lora: Any) -> Any:
    """Run one blocking vLLM batch off the event loop, so the reusable replica keeps heartbeating."""
    try:
        return await asyncio.to_thread(engine.generate, prompts, sampling, lora_request=lora)
    except Exception as e:
        if type(e).__name__ == "EngineDeadError":
            # The replica outlives this task and serves other runs; don't hand them the dead engine.
            _load_engine.cache_clear()
        raise


# {{docs-fragment lora_request}}
async def _lora_request(adapter: flyte.io.Dir | None, version: int) -> Any:
    if adapter is None:
        return None
    import zlib

    from vllm.lora.request import LoRARequest

    # vLLM caches adapters by lora_int_id, so it must be >= 1 and unique per adapter *across runs*: the
    # engine is shared, and every run has its own v1, v2, ... Hash the remote path, not the version.
    # vLLM stores the id in an int32 array, so fold the hash into [1, 2**31 - 1].
    lora_id = zlib.crc32(adapter.path.encode()) % (2**31 - 1) + 1
    return LoRARequest(f"policy-v{version}-{lora_id:08x}", lora_id, await _local_dir(adapter.path))


# {{/docs-fragment lora_request}}


# {{docs-fragment generate}}
@rollout_env.task
async def generate(
    base: flyte.io.Dir,
    task: str,
    questions: list[str],
    answers: list[str],
    adapter: flyte.io.Dir,
    version: int,
    first_group: int,
    max_new_tokens: int,
) -> list[Rollout]:
    """Sample GROUP_SIZE completions for each question in one vLLM batch, using the current adapter."""
    from vllm import SamplingParams

    engine = await _load_engine(base.path)
    tokenizer = engine.get_tokenizer()
    prompts = [_render(tokenizer, task, q) for q in questions]
    sampling = SamplingParams(n=GROUP_SIZE, temperature=SAMPLING_TEMPERATURE, top_p=1.0, max_tokens=max_new_tokens)
    lora = await _lora_request(adapter, version)
    outputs = await _generate(engine, prompts, sampling, lora)
    return [
        Rollout(
            group_id=first_group + i,
            question=q,
            answer=a,
            completion=o.text,
            completion_ids=list(o.token_ids),
        )
        for i, (q, a, out) in enumerate(zip(questions, answers, outputs))
        for o in out.outputs
    ]


# {{/docs-fragment generate}}


# {{docs-fragment evaluate}}
@rollout_env.task
async def evaluate(
    base: flyte.io.Dir,
    task: str,
    max_new_tokens: int,
    adapter: flyte.io.Dir | None = None,
    version: int = 0,
    sample_index: int = -1,
) -> EvalResult:
    """Greedy accuracy on the fixed held-out set (base model when ``adapter`` is None).

    The sample shown in the report is ``sample_index`` if given, otherwise the first problem the model
    got wrong — so the base eval picks a failure and the final eval shows the same problem again.
    """
    from vllm import SamplingParams

    engine = await _load_engine(base.path)
    tokenizer = engine.get_tokenizer()
    problems = tasks.eval_set(task, EVAL_SIZE)
    prompts = [_render(tokenizer, task, q) for q, _ in problems]
    sampling = SamplingParams(n=1, temperature=0.0, max_tokens=max_new_tokens)
    lora = await _lora_request(adapter, version)
    outputs = await _generate(engine, prompts, sampling, lora)
    completions = [out.outputs[0].text for out in outputs]
    graded = [tasks.score(task, c, a) for c, (_, a) in zip(completions, problems)]

    if sample_index < 0:
        sample_index = next((i for i, (_, _, ok) in enumerate(graded) if not ok), 0)
    n = len(problems)
    result = EvalResult(
        accuracy=sum(ok for _, _, ok in graded) / n,
        format_rate=sum(fmt for _, fmt, _ in graded) / n,
        num_problems=n,
        sample_index=sample_index,
        sample_question=problems[sample_index][0],
        sample_completion=completions[sample_index],
        sample_correct=graded[sample_index][2],
    )
    logger.info(
        "eval %s (adapter=%s): accuracy %.1f%%",
        task,
        "none" if adapter is None else f"v{version}",
        100 * result.accuracy,
    )
    return result


# {{/docs-fragment evaluate}}


# ----------------------------------------------------------------------------------------------------
# 2. Reward — verifiable, rule-based (tasks.score)
# ----------------------------------------------------------------------------------------------------
# This task is the only place the loop learns what "good" means. Here the reward is verifiable (RLVR):
# tasks.score runs a checker and returns a number. To train on human feedback instead (RLHF), keep the
# signature and return a learned reward model's score, a judge model's rating, or stored human ratings.
# A reward that needs a GPU just needs its own TaskEnvironment with one; the driver doesn't change.
# {{docs-fragment reward}}
@reward_env.task
async def score_chunk(task: str, rollouts: list[Rollout]) -> list[float]:
    """Score one generate() chunk; runs while the other chunk is still generating."""
    return [tasks.score(task, r.completion, r.answer)[0] for r in rollouts]


# {{/docs-fragment reward}}


# ----------------------------------------------------------------------------------------------------
# 3. Trainer — GRPO step on a LoRA adapter, model + optimizer kept warm across steps
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment init_adapter}}
@train_env.task
async def init_adapter(base: flyte.io.Dir) -> flyte.io.Dir:
    """Create a fresh (zero-effect) LoRA adapter so iteration 0 has something to attach."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(await _local_dir(base.path), dtype=torch.bfloat16)
    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )
    out_dir = tempfile.mkdtemp(prefix="adapter-v0-")
    get_peft_model(model, config).save_pretrained(out_dir)
    return await flyte.io.Dir.from_local(out_dir)


# {{/docs-fragment init_adapter}}


# {{docs-fragment trainer_state}}
# In-memory trainer state for this warm replica, shared by every run that lands on it:
#   base, model, tokenizer - one frozen base on the GPU
#   policies[skill]        - that skill's LoRA adapter (a named PEFT adapter on the shared base), its
#                            AdamW state, and the remote path the in-memory weights correspond to
# A step whose input adapter matches the recorded path skips the reload and keeps Adam's moments.
_TRAINER: dict[str, Any] = {}
# Held while _TRAINER is being (re)loaded and for the GPU part of a step.
_TRAINER_LOCK = asyncio.Lock()


async def _trainer(base: flyte.io.Dir, task: str, adapter: flyte.io.Dir) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(shared state, this skill's policy)`` with the skill's adapter active and trainable."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    st = _TRAINER
    if st.get("base") != base.path:
        logger.info("Loading base model %s", base.path)
        st.clear()
        torch.cuda.empty_cache()
        local_base = await _local_dir(base.path)
        st.update(
            base=base.path,
            model=AutoModelForCausalLM.from_pretrained(local_base, dtype=torch.bfloat16).to("cuda"),
            tokenizer=AutoTokenizer.from_pretrained(local_base),
            policies={},
        )

    policy = st["policies"].get(task)
    if policy is not None and policy["adapter"] == adapter.path:
        st["model"].set_adapter(task)  # another skill's step may have run in between
        return st, policy

    logger.info("Loading %s adapter %s", task, adapter.path)
    st["policies"].pop(task, None)
    local_adapter = await adapter.download()
    model = st["model"]
    if not isinstance(model, PeftModel):  # first adapter on this replica: wrap the base
        model = st["model"] = PeftModel.from_pretrained(model, local_adapter, adapter_name=task, is_trainable=True)
    else:
        if task in model.peft_config:  # stale weights from an earlier run of the same skill
            model.delete_adapter(task)
        model.load_adapter(local_adapter, adapter_name=task, is_trainable=True)
    model.set_adapter(task)  # activates this adapter and freezes every other one
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"adapter {task!r} has no trainable parameters")
    policy = st["policies"][task] = {
        "adapter": adapter.path,
        "params": params,
        "optimizer": torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0),
    }
    return st, policy


# {{/docs-fragment trainer_state}}


# {{docs-fragment advantages}}
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


# {{/docs-fragment advantages}}


# {{docs-fragment grpo_step}}
def _grpo_step(
    st: dict[str, Any], policy: dict[str, Any], task: str, rollouts: list[Rollout], rewards: list[float], version: int
) -> tuple[pathlib.Path, pathlib.Path, float, int]:
    """The blocking GPU part of a step: backward passes, optimizer step, adapter saved to local disk.

    Returns ``(tmp_dir, adapter_dir, mean_loss, contributing)``. Call with ``_TRAINER_LOCK`` held.
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer, optimizer = st["model"], st["tokenizer"], policy["optimizer"]

    advantages = _group_normalized_advantages(rollouts, rewards)
    items = [(r, adv) for r, adv in zip(rollouts, advantages) if adv != 0.0 and r.completion_ids]
    prompt_ids = {
        q: tokenizer(_render(tokenizer, task, q), add_special_tokens=False).input_ids
        for q in {r.question for r, _ in items}
    }

    optimizer.zero_grad()
    total_loss = 0.0
    for start in range(0, len(items), MICRO_BATCH):
        batch = items[start : start + MICRO_BATCH]
        seqs = [prompt_ids[r.question] + r.completion_ids for r, _ in batch]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        loss_mask = torch.zeros((len(seqs), width), dtype=torch.float32)
        for row, ((r, _), seq) in enumerate(zip(batch, seqs)):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention[row, : len(seq)] = 1
            loss_mask[row, len(prompt_ids[r.question]) : len(seq)] = 1.0  # completion tokens only
        input_ids, attention, loss_mask = input_ids.cuda(), attention.cuda(), loss_mask.cuda()

        logits = model(input_ids=input_ids, attention_mask=attention).logits[:, :-1].float()
        token_logp = -F.cross_entropy(logits.transpose(1, 2), input_ids[:, 1:], reduction="none")
        mask = loss_mask[:, 1:]
        seq_logp = (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        adv = torch.tensor([a for _, a in batch], device=seq_logp.device)
        # Mean over every contributing sequence in the step, so the step size doesn't scale with batch.
        loss = -(adv * seq_logp).sum() / len(items)
        loss.backward()
        total_loss += float(loss.item())

    if items:
        torch.nn.utils.clip_grad_norm_(policy["params"], 1.0)
        optimizer.step()
    logger.info("GRPO step v%d: %d/%d rollouts contributed, loss %.4f", version, len(items), len(rollouts), total_loss)

    out_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"adapter-{task}-v{version}-"))
    model.save_pretrained(str(out_dir), selected_adapters=[task])
    # PEFT writes a named (non-"default") adapter into a subdirectory named after it.
    saved = out_dir / task if (out_dir / task / "adapter_config.json").exists() else out_dir
    return out_dir, saved, total_loss, len(items)


# {{/docs-fragment grpo_step}}


# {{docs-fragment train_step}}
@train_env.task
async def train_step(
    base: flyte.io.Dir,
    task: str,
    rollouts: list[Rollout],
    rewards: list[float],
    adapter: flyte.io.Dir,
    version: int,
) -> tuple[flyte.io.Dir, float, int]:
    """One GRPO policy-gradient step; returns ``(new_adapter, mean_loss, contributing)``."""
    async with _TRAINER_LOCK:
        st, policy = await _trainer(base, task, adapter)
        # From here the in-memory weights match no published adapter until the upload below finishes;
        # if this step fails midway, the retry reloads from ``adapter`` instead of training on top.
        policy["adapter"] = None
        # Off the event loop, so the replica keeps heartbeating and the other skill's I/O keeps moving.
        out_dir, saved, mean_loss, contributing = await asyncio.to_thread(
            _grpo_step, st, policy, task, rollouts, rewards, version
        )
    try:
        new_adapter = await flyte.io.Dir.from_local(str(saved))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)  # don't fill the long-lived replica's disk
    policy["adapter"] = new_adapter.path  # the next step resumes from memory instead of reloading
    return new_adapter, mean_loss, contributing


# {{/docs-fragment train_step}}


# {{docs-fragment merge_adapter}}
@train_env.task
async def merge_adapter(base: flyte.io.Dir, adapter: flyte.io.Dir) -> flyte.io.Dir:
    """Merge the LoRA into the base: a plain HF checkpoint that vLLM can stream with no adapter overhead."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_base = await _local_dir(base.path)
    model = AutoModelForCausalLM.from_pretrained(local_base, dtype=torch.bfloat16)  # CPU is plenty
    merged = PeftModel.from_pretrained(model, await adapter.download()).merge_and_unload()
    out_dir = tempfile.mkdtemp(prefix="merged-")
    merged.save_pretrained(out_dir, safe_serialization=True)  # weights + config + generation_config
    AutoTokenizer.from_pretrained(local_base).save_pretrained(out_dir)  # chat template included
    return await flyte.io.Dir.from_local(out_dir)


# {{/docs-fragment merge_adapter}}


# {{docs-fragment model_artifact}}
async def _as_model_artifact(model: flyte.io.Dir, task: str, base_accuracy: float, final_accuracy: float) -> Any:
    """Wrap a merged model Dir so the run publishes it as ``MODEL_ARTIFACTS[task]``, with a model card.

    The task returning it must be declared with ``produces_artifacts=True``; without that flag the backend
    ignores the metadata and silently records no artifact.
    """
    from flyte import artifacts

    summary = (
        f"# {MODEL_ARTIFACTS[task]}\n\n"
        f"`{BASE_MODEL_REPO}` trained with GRPO + LoRA (rank {LORA_RANK}) on the `{task}` task, "
        f"adapter merged into the base weights.\n\n"
        f"| | Held-out accuracy ({EVAL_SIZE} problems) |\n|---|---|\n"
        f"| Base model | {base_accuracy:.0%} |\n| After RL | {final_accuracy:.0%} |\n\n"
        f"System prompt used in training:\n\n```\n{tasks.SYSTEM_PROMPTS[task]}\n```\n"
    )
    card = None
    try:
        card = await artifacts.Card.create_from.aio(content=summary, format="md", card_type="model")
    except Exception as e:  # noqa: BLE001 - the artifact is still useful without its card
        logger.warning("Could not upload model card: %s", e)

    metadata = artifacts.Metadata.create_model_metadata(
        name=MODEL_ARTIFACTS[task],
        description=f"{BASE_MODEL_REPO} RL-trained on {task}: {base_accuracy:.0%} -> {final_accuracy:.0%}",
        card=card,
        framework="huggingface",
        model_type="qwen2",
        architecture="Qwen2ForCausalLM",
        task="text-generation",
        attrs={
            "base_model": BASE_MODEL_REPO,
            "rl_task": task,
            "base_accuracy": f"{base_accuracy:.4f}",
            "final_accuracy": f"{final_accuracy:.4f}",
        },
    )
    return artifacts.new(model, metadata)


# {{/docs-fragment model_artifact}}


# {{docs-fragment review}}
async def _review(task: str, base_eval: EvalResult, final_eval: EvalResult) -> bool:
    """Pause this run until a person approves or rejects the trained model. Returns True to publish."""
    approval = await flyte.new_condition.aio(
        f"approve-{task}",
        prompt=(
            f"## Publish the `{task}` model?\n\n"
            f"| | Held-out accuracy ({final_eval.num_problems} problems) |\n|---|---|\n"
            f"| Base model | {base_eval.accuracy:.0%} |\n"
            f"| After RL | {final_eval.accuracy:.0%} |\n\n"
            f"The same held-out problem, before and after:\n\n"
            f"```\n{final_eval.sample_question}\n```\n\n"
            f"Base model ({'correct' if base_eval.sample_correct else 'wrong'}):\n\n"
            f"```\n{base_eval.sample_completion[:1500]}\n```\n\n"
            f"After RL ({'correct' if final_eval.sample_correct else 'wrong'}):\n\n"
            f"```\n{final_eval.sample_completion[:1500]}\n```\n\n"
            f"Approve to merge the adapter and publish `{MODEL_ARTIFACTS[task]}`. "
            "The reward curves are in this run's report."
        ),
        prompt_type="markdown",
        data_type=bool,
        timeout=REVIEW_TIMEOUT,
    )
    approved = await approval.wait.aio()
    logger.info("review %s: %s", task, "approved" if approved else "rejected")
    return approved


# {{/docs-fragment review}}


# ----------------------------------------------------------------------------------------------------
# 4. Driver — the RL loop
# ----------------------------------------------------------------------------------------------------
async def _publish_report(
    history: list[IterationMetrics],
    task: str,
    num_iterations: int,
    status: str,
    base_eval: EvalResult | None,
    final_eval: EvalResult | None = None,
) -> None:
    html = render_report(
        history,
        task=task,
        base_model=BASE_MODEL_REPO,
        num_iterations=num_iterations,
        group_size=GROUP_SIZE,
        prompts_per_iter=PROMPTS_PER_ITER,
        lora_rank=LORA_RANK,
        learning_rate=LEARNING_RATE,
        status=status,
        base_eval=base_eval,
        final_eval=final_eval,
    )
    await flyte.report.replace.aio(html, do_flush=True)


# {{docs-fragment train_rl}}
@driver_env.task(report=True, produces_artifacts=True)
async def train_rl(base: flyte.io.Dir, skill: str = "countdown", num_iterations: int = 0) -> flyte.io.Dir:
    """Eval the base, run the GRPO loop, eval again, wait for a review, and publish the merged model.

    ``skill`` is one of ``tasks.TASKS`` (not named ``task``: that would clash with ``flyte.run``'s first
    parameter). Loop state is checkpointed every iteration, so a preempted driver resumes mid-run.
    """
    task = skill
    if task not in tasks.TASKS:
        raise ValueError(f"task must be one of {tasks.TASKS}, got {task!r}")
    cfg = TASK_CONFIG[task]
    num_iterations = num_iterations or cfg.iterations

    ctx = flyte.ctx()
    cp = ctx.checkpoint if ctx is not None else None
    start_iter = 0
    adapter: flyte.io.Dir | None = None
    adapter_version = 0
    history: list[IterationMetrics] = []
    base_eval: EvalResult | None = None

    if cp is not None:
        prev = await cp.load()
        if prev is not None:
            state = json.loads(prev.read_text())
            start_iter = state["iteration"] + 1
            adapter_version = state["adapter_version"]
            adapter = flyte.io.Dir.from_existing_remote(state["adapter_path"])
            history = [IterationMetrics(**row) for row in state["history"]]
            base_eval = EvalResult(**state["base_eval"])
            logger.info("Resumed at iteration %d (adapter v%d)", start_iter, adapter_version)

    # Act 1: how bad is the base model?
    if base_eval is None:
        with flyte.group("eval-base"):
            base_eval = await evaluate(base, task, cfg.max_new_tokens)
    await _publish_report(history, task, num_iterations, "running", base_eval)

    if adapter is None:
        adapter = await init_adapter(base)

    # Act 2: RL.
    for it in range(start_iter, num_iterations):
        with flyte.group(f"iter-{it}"):
            problems = tasks.train_batch(task, it, PROMPTS_PER_ITER)
            per_chunk = -(-len(problems) // ROLLOUT_CHUNKS)
            gen_futs = [
                asyncio.create_task(
                    generate(
                        base,
                        task,
                        [q for q, _ in problems[s : s + per_chunk]],
                        [a for _, a in problems[s : s + per_chunk]],
                        adapter,
                        adapter_version,
                        first_group=s,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                )
                for s in range(0, len(problems), per_chunk)
            ]
            # Score each chunk as soon as it lands, overlapping reward with the other chunk's generation.
            rollouts: list[Rollout] = []
            score_futs: list[asyncio.Task[list[float]]] = []
            for fut in asyncio.as_completed(gen_futs):
                chunk = await fut
                rollouts.extend(chunk)
                score_futs.append(asyncio.create_task(score_chunk(task, chunk)))
            rewards = [r for chunk_rewards in await asyncio.gather(*score_futs) for r in chunk_rewards]

            adapter, mean_loss, contributing = await train_step(
                base, task, rollouts, rewards, adapter, adapter_version + 1
            )
            adapter_version += 1

            graded = [tasks.score(task, r.completion, r.answer) for r in rollouts]
            n = len(rollouts)
            best = max(range(n), key=lambda i: rewards[i])
            history.append(
                IterationMetrics(
                    iteration=it,
                    adapter_version=adapter_version,
                    num_rollouts=n,
                    mean_reward=sum(rewards) / n,
                    max_reward=max(rewards),
                    accuracy=sum(ok for _, _, ok in graded) / n,
                    format_rate=sum(fmt for _, fmt, _ in graded) / n,
                    mean_loss=mean_loss,
                    contributing=contributing,
                    sample_question=rollouts[best].question,
                    sample_completion=rollouts[best].completion,
                    sample_reward=rewards[best],
                )
            )
            logger.info(
                "iter %d: mean reward %.3f, accuracy %.1f%%", it, history[-1].mean_reward, 100 * history[-1].accuracy
            )
            await _publish_report(history, task, num_iterations, "running", base_eval)

            if cp is not None:
                state = {
                    "iteration": it,
                    "adapter_version": adapter_version,
                    "adapter_path": adapter.path,
                    "history": [vars(m) for m in history],
                    "base_eval": vars(base_eval),
                }
                await cp.save(json.dumps(state).encode())

    # Act 3: same held-out problems, trained model.
    with flyte.group("eval-final"):
        final_eval = await evaluate(
            base, task, cfg.max_new_tokens, adapter, adapter_version, sample_index=base_eval.sample_index
        )

    # Act 4: a person decides whether this skill's model ships. The GPUs are not held while the run waits:
    # the warm replicas keep serving the other runs, and scale down once every run is idle.
    await _publish_report(history, task, num_iterations, "review", base_eval, final_eval)
    if not await _review(task, base_eval, final_eval):
        await _publish_report(history, task, num_iterations, "rejected", base_eval, final_eval)
        raise RuntimeError(f"The {task} model was rejected in review; nothing was published.")

    # Approved: merge the adapter into the base weights and publish the model artifact.
    model = await merge_adapter(base, adapter)
    await _publish_report(history, task, num_iterations, "complete", base_eval, final_eval)
    return await _as_model_artifact(model, task, base_eval.accuracy, final_eval.accuracy)


# {{/docs-fragment train_rl}}


@driver_env.task(produces_artifacts=True)
async def publish_model(model: flyte.io.Dir, skill: str, base_accuracy: float, final_accuracy: float) -> flyte.io.Dir:
    """(Re)publish an already merged model as the task's artifact, without retraining."""
    return await _as_model_artifact(model, skill, base_accuracy, final_accuracy)


# ----------------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------------
# {{docs-fragment entrypoint}}
def _base_model() -> Any:
    """The prefetched base model artifact; prefetch it on first use."""
    import flyte.prefetch
    import flyte.remote

    existing = next(iter(flyte.remote.Artifact.listall(name=BASE_ARTIFACT, limit=1)), None)
    if existing is not None:
        return existing
    run = flyte.prefetch.hf_model(repo=BASE_MODEL_REPO, hf_token_key=None)  # public repo, no HF token
    print(f"Prefetching {BASE_MODEL_REPO}: {run.url}")
    run.wait()
    return flyte.remote.Artifact.get(BASE_ARTIFACT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task",
        choices=(*tasks.TASKS, "all"),
        default="countdown",
        help="skill to train; 'all' launches one run per skill, sharing the same warm GPU replicas",
    )
    parser.add_argument("--iterations", type=int, default=0, help="GRPO iterations (default: per-task)")
    args = parser.parse_args()

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    base_model = _base_model()
    for skill in tasks.TASKS if args.task == "all" else (args.task,):
        run = flyte.run(train_rl, base=base_model, skill=skill, num_iterations=args.iterations)
        print(f"{skill}: {run.url}")
        print(f"  when it pauses for review: approve in the UI, or python review.py {run.name} --approve")
# {{/docs-fragment entrypoint}}
