# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""RL on LAB tasks with the loss Harvey describes, on one 8-GPU node, to make checkpoints for the eval matrix.

What matches Harvey's published recipe: rank-64 LoRA over every linear layer, CISPO (the importance weight is
clipped to [0.8, 1.2] and carries no gradient) with Dr. GRPO's advantage (reward minus the group mean, no std) and a
constant loss normalizer (T0 = 55,000 tokens), G = 8 rollouts per task, only model-generated tokens in the loss, and
the reward ``0.35 c + 0.35 h + 0.30 a`` with zero for a cut-off episode.

What doesn't: it's synchronous (sample with the current adapter, grade, one optimizer step, repeat), so the
importance weight corrects train/inference numerics, not staleness; it's a 7B model, not 120B-816B; and the episodes
are capped at a few dozen turns, not 200. This is a way to get checkpoints to measure, not a replication.

    GPUs 0-1   judge (gpt-oss-120b, data-parallel x2)
    GPUs 2-5   policy sampler (vLLM, data-parallel x4, LoRA hot-loaded each step)
    GPU  6     trainer (PyTorch + PEFT)

    CLUSTER=amd GPU_QUEUE=<queue> GPU_IMAGE_URI=<uri> python train.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import statistics as st
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.report
from pydantic import BaseModel, Field

import lab_harness as lh
import servers
from config import CLUSTER, PROPAGATE, gpu_env_kwargs
from images import LAB_COMMIT

train_env = flyte.TaskEnvironment(name="legal-rl-train", env_vars=PROPAGATE, **gpu_env_kwargs())


# {{docs-fragment config}}
class TrainConfig(BaseModel):
    policy_hf: str = "openai/gpt-oss-20b"
    reasoning_effort: str = "medium"
    attn_implementation: str = "flex_attention"  # gpt-oss's attention sinks rule out plain SDPA; eager is O(n^2) memory
    max_train_len: int = 40_000
    # gpt-oss's template drops earlier reasoning, so every turn is its own sequence. Train on this many turns per
    # rollout, sampled at random and weighted by (turns / sampled) so the expected gradient is the full one.
    turns_per_rollout: int = 4  # longer sequences are cut from the left of the prompt, never inside the model's tokens
    train_tasks: list[str]
    steps: int = 30
    groups_per_step: int = 8  # N task groups per optimizer step (Harvey: 16)
    group_size: int = 8  # G rollouts per task (Harvey: 8)
    lr: float = 5e-5  # Harvey swept [1e-5, 1e-4] on their smallest model
    lora_rank: int = 64
    lora_alpha: int = 64
    clip_low: float = 0.8  # CISPO importance-weight clip, as published
    clip_high: float = 1.2
    t0: int = 55_000  # Dr. GRPO's constant loss normalizer, as published
    weights: lh.RewardWeights = lh.RewardWeights()
    harness: lh.Harness = lh.Harness(max_turns=30)
    temperature: float = 1.0
    max_tokens: int = 4096
    judge_hf: str = "openai/gpt-oss-120b"
    checkpoint_every: int = 10
    seed: int = 0
# {{/docs-fragment config}}


class StepLog(BaseModel):
    step: int
    mean_reward: float
    criteria_pass: float
    all_pass: float
    truncated: float
    no_deliverable: float
    mean_turns: float
    mean_output_tokens: float
    deliverable_chars: float
    groups_with_signal: int
    train_tokens: int
    samples: int
    loss: float
    grad_norm: float
    ratio_mean: float
    ratio_clipped: float
    rollout_s: float
    grade_s: float
    train_s: float


class TrainResult(BaseModel):
    checkpoints: dict[int, flyte.io.Dir]  # step -> LoRA adapter directory (step 0 is the base model: no adapter)
    history: list[StepLog]
    config: TrainConfig
    meta: dict = Field(default_factory=dict)


# --- turns -> training sequences --------------------------------------------------------------------------------


class Sample(BaseModel):
    ids: list[int]
    mask: list[int]  # 1 on model-generated tokens, 0 on prompt and tool results
    old_logprobs: list[float]  # aligned with ids; 0 where mask is 0
    advantage: float


# {{docs-fragment merge}}
def to_samples(turns: list[lh.TurnTokens], advantage: float) -> list[Sample]:
    """One training sequence per run of turns whose prompts extend each other token for token.

    Each turn's prompt is the whole conversation so far, as the server rendered it. When turn t's prompt starts with
    turn t-1's prompt plus its completion, the two are one sequence with two masked spans; when the chat template
    rewrites history (some drop earlier reasoning), the turn starts a new sequence. Either way the loss sees exactly
    the tokens the sampler produced, in exactly the context it produced them.
    """
    out: list[Sample] = []
    ids: list[int] = []
    mask: list[int] = []
    lps: list[float] = []
    for t in turns:
        if not t.completion_ids:
            continue
        if ids and t.prompt_ids[:len(ids)] == ids:
            extra = t.prompt_ids[len(ids):]
        else:
            if ids:
                out.append(Sample(ids=ids, mask=mask, old_logprobs=lps, advantage=advantage))
            ids, mask, lps, extra = [], [], [], t.prompt_ids
        n = min(len(t.completion_ids), len(t.logprobs))
        ids += extra + t.completion_ids[:n]
        mask += [0] * len(extra) + [1] * n
        lps += [0.0] * len(extra) + t.logprobs[:n]
    if ids:
        out.append(Sample(ids=ids, mask=mask, old_logprobs=lps, advantage=advantage))
    return out
# {{/docs-fragment merge}}


def _clip_left(s: Sample, n: int) -> Sample:
    """Drop the oldest context so the sequence fits, keeping every model token that's still predictable."""
    if len(s.ids) <= n:
        return s
    cut = len(s.ids) - n
    mask = s.mask[cut:]
    mask[0] = 0  # the first kept token has no context to be predicted from
    return Sample(ids=s.ids[cut:], mask=mask, old_logprobs=s.old_logprobs[cut:], advantage=s.advantage)


# --- the trainer ----------------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: TrainConfig):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        self.torch, self.cfg = torch, cfg
        kw = {}
        if "gpt-oss" in cfg.policy_hf:  # MXFP4 experts: dequantize to bf16 for training
            from transformers import Mxfp4Config

            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.policy_hf, dtype=torch.bfloat16,
                                                     attn_implementation=cfg.attn_implementation, **kw).to("cuda")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        # "LoRA over the full network": every linear layer, plus the fused expert weights of a MoE.
        experts = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if "gpt-oss" in cfg.policy_hf else None
        self.model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                                      target_modules="all-linear", target_parameters=experts,
                                                      lora_dropout=0.0))
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr,
                                     betas=(0.9, 0.99), weight_decay=0.0)

    # {{docs-fragment cispo}}
    def step(self, samples: list[Sample], n_groups: int) -> dict:
        """One optimizer step. CISPO: loss = -sum_t sg(clip(r_t, lo, hi)) * A * log pi(t), over model tokens only,
        divided by (groups x G x T0) as in Dr. GRPO, so a long rollout's tokens weigh what a short one's do."""
        torch, cfg = self.torch, self.cfg
        self.model.train()
        denom = n_groups * cfg.group_size * cfg.t0
        ratios, clipped, total_loss, tokens = [], 0, 0.0, 0
        for s in samples:
            s = _clip_left(s, cfg.max_train_len)
            ids = torch.tensor([s.ids], device="cuda")
            pos = torch.tensor([i - 1 for i, m in enumerate(s.mask) if m], device="cuda")  # logits that predict them
            tgt = ids[0, pos + 1]
            hidden = self.model.base_model.model.model(input_ids=ids).last_hidden_state[0, pos]
            logp = torch.log_softmax(self.model.base_model.model.lm_head(hidden).float(), -1).gather(-1, tgt[:, None])[:, 0]
            old = torch.tensor([s.old_logprobs[i] for i, m in enumerate(s.mask) if m], device="cuda")
            ratio = torch.exp(logp.detach() - old)
            w = ratio.clamp(cfg.clip_low, cfg.clip_high)
            loss = -(w * s.advantage * logp).sum() / denom
            loss.backward()
            total_loss += float(loss)
            ratios.append(float(ratio.mean()))
            clipped += int(((ratio < cfg.clip_low) | (ratio > cfg.clip_high)).sum())
            tokens += int(pos.numel())
        gn = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {"loss": total_loss, "grad_norm": float(gn), "ratio_mean": st.fmean(ratios) if ratios else 1.0,
                "ratio_clipped": clipped / max(tokens, 1), "train_tokens": tokens}
    # {{/docs-fragment cispo}}

    def save(self, path: str) -> str:
        self.model.save_pretrained(path)
        return path


# --- the loop -------------------------------------------------------------------------------------------------


async def _rollouts(lab: Path, tasks: list[str], policy: lh.Policy, harness: lh.Harness, url: str, g: int,
                    step: int) -> list[lh.Rollout]:
    sem = asyncio.Semaphore(96)

    async def one(t, k):
        async with sem:
            try:
                return await asyncio.to_thread(lh.run_rollout, lab, t, policy, harness, url, step * 100 + k, True)
            except Exception as e:  # noqa: BLE001  a crashed episode is a zero-reward episode, not a crashed step
                return lh.Rollout(rollout_id=f"err-{t}-{k}", task_id=t, policy=policy.name, harness=harness.version,
                                  sample=k, finish_reason="error", turns=0, input_tokens=0, output_tokens=0, wall_s=0,
                                  error=f"{type(e).__name__}: {e}"[:300])

    return await asyncio.gather(*[one(t, k) for t in tasks for k in range(g)])


async def _grade(lab: Path, rollouts: list[lh.Rollout], judge: lh.OpenAICompatJudge, groups: dict) -> list[dict]:
    sem = asyncio.Semaphore(32)

    async def one(r):
        if r.finish_reason == "error" or not r.outputs_b64:
            return {"c": 0.0, "h": 0.0, "a": 0.0}
        async with sem:
            task, _ = lh.load_task(lab, r.task_id)
            v = await asyncio.to_thread(lh.grade, r, task, judge, judge.model, 16)
        return lh.reward_terms(v.verdicts, groups.get(r.task_id, {}))

    return await asyncio.gather(*[one(r) for r in rollouts])


@flyte.trace
async def traced_groups(lab_root: str, task_id: str, model: str, base_url: str) -> dict[str, list[str]]:
    import openai

    task, _ = lh.load_task(Path(lab_root), task_id)
    return await asyncio.to_thread(lh.group_issues, task, openai.OpenAI(base_url=base_url, api_key="x"), model)


# {{docs-fragment train}}
@train_env.task(cache="auto", retries=2, report=True)
async def train(cfg: TrainConfig) -> TrainResult:
    os.environ[servers.VISIBLE] = "6" if CLUSTER == "amd" else "0"  # the trainer's GPU, before torch touches one
    t_start = time.time()
    lab = lh.fetch_lab(cfg.train_tasks, LAB_COMMIT, Path("/tmp/lab"))
    amd = CLUSTER == "amd"
    judge_srv = servers.ServerSpec(served_name="judge", hf_id=cfg.judge_hf, gpus=[0, 1] if amd else [0],
                                   max_model_len=131072, gpu_memory_utilization=0.85 if amd else 0.3)
    pol_srv = servers.ServerSpec(served_name="policy", hf_id=cfg.policy_hf, gpus=[2, 3, 4, 5] if amd else [0],
                                 max_model_len=131072, enable_lora=True, max_lora_rank=cfg.lora_rank,
                                 gpu_memory_utilization=0.85 if amd else 0.3, port=8300)
    await asyncio.gather(servers.ensure(judge_srv), servers.ensure(pol_srv))
    judge = lh.OpenAICompatJudge(lh.JudgeSpec(name="judge", hf_id=cfg.judge_hf, reasoning_effort="low"), judge_srv.url)
    groups = dict(zip(cfg.train_tasks, await asyncio.gather(*[
        traced_groups(str(lab), t, "judge", judge_srv.url) for t in cfg.train_tasks])))

    trainer = Trainer(cfg)
    rng = random.Random(cfg.seed)
    history: list[StepLog] = []
    ckpts: dict[int, flyte.io.Dir] = {}
    adapter = "policy"  # step 0 samples from the base model
    for step in range(cfg.steps + 1):
        if step % cfg.checkpoint_every == 0 and step > 0:
            d = trainer.save(f"/tmp/ckpt/step-{step:04d}")
            ckpts[step] = await flyte.io.Dir.from_local(d)
        if step == cfg.steps:
            break
        tasks = rng.sample(cfg.train_tasks, cfg.groups_per_step)
        policy = lh.Policy(name=adapter, hf_id=cfg.policy_hf, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                           reasoning_effort=cfg.reasoning_effort)
        t0 = time.time()
        rollouts = await _rollouts(lab, tasks, policy, cfg.harness, pol_srv.url, cfg.group_size, step)
        t1 = time.time()
        terms = await _grade(lab, rollouts, judge, groups)
        t2 = time.time()
        rewards = [lh.reward(x, r.finish_reason, cfg.weights) if r.finish_reason != "error" else 0.0
                   for x, r in zip(terms, rollouts)]
        samples, live = [], 0
        for gi in range(cfg.groups_per_step):
            sl = slice(gi * cfg.group_size, (gi + 1) * cfg.group_size)
            rs, Rs = rollouts[sl], rewards[sl]
            mean = st.fmean(Rs)
            if max(Rs) - min(Rs) < 1e-9:
                continue  # zero advantage for every rollout: no gradient, skip the compute
            live += 1
            for r, R in zip(rs, Rs):
                ss = to_samples(r.tokens, R - mean)
                if len(ss) > cfg.turns_per_rollout:
                    w = len(ss) / cfg.turns_per_rollout
                    ss = [x.model_copy(update={"advantage": x.advantage * w})
                          for x in rng.sample(ss, cfg.turns_per_rollout)]
                samples += ss
        stats = trainer.step(samples, cfg.groups_per_step) if samples else {
            "loss": 0.0, "grad_norm": 0.0, "ratio_mean": 1.0, "ratio_clipped": 0.0, "train_tokens": 0}
        t3 = time.time()
        adapter = f"step-{step + 1:04d}"
        path = trainer.save(f"/tmp/adapters/{adapter}")
        await servers.load_lora(pol_srv, adapter, path)
        ok = [r for r in rollouts if r.finish_reason != "error"]
        log = StepLog(step=step, mean_reward=st.fmean(rewards), criteria_pass=st.fmean(x["c"] for x in terms),
                      all_pass=st.fmean(x["a"] for x in terms),
                      truncated=st.fmean(r.finish_reason in lh.TRUNCATED for r in rollouts),
                      no_deliverable=st.fmean(r.deliverable_chars == 0 for r in rollouts),
                      mean_turns=st.fmean(r.turns for r in ok) if ok else 0,
                      mean_output_tokens=st.fmean(r.output_tokens for r in ok) if ok else 0,
                      deliverable_chars=st.fmean(r.deliverable_chars for r in ok) if ok else 0,
                      groups_with_signal=live, samples=len(samples), rollout_s=t1 - t0, grade_s=t2 - t1,
                      train_s=t3 - t2, **stats)
        history.append(log)
        print(log.model_dump_json(), flush=True)
        await flyte.report.replace.aio(_report(history, time.time() - t_start), do_flush=True)
    return TrainResult(checkpoints=ckpts, history=history, config=cfg,
                       meta={"minutes": round((time.time() - t_start) / 60, 1), "cluster": CLUSTER})
# {{/docs-fragment train}}


def _report(history: list[StepLog], elapsed: float) -> str:
    rows = "".join(f"<tr><td>{h.step}</td><td>{h.mean_reward:.3f}</td><td>{h.criteria_pass:.3f}</td>"
                   f"<td>{h.truncated:.2f}</td><td>{h.mean_output_tokens:.0f}</td><td>{h.deliverable_chars:.0f}</td>"
                   f"<td>{h.groups_with_signal}</td><td>{h.ratio_mean:.3f}</td><td>{h.rollout_s + h.grade_s + h.train_s:.0f}</td></tr>"
                   for h in history)
    return (f"<h3>training: {len(history)} steps, {elapsed / 60:.0f} min</h3><table border=1 cellpadding=4>"
            "<tr><th>step</th><th>reward</th><th>criteria</th><th>truncated</th><th>output tokens</th>"
            f"<th>deliverable bytes</th><th>groups w/ signal</th><th>IS ratio</th><th>s</th></tr>{rows}</table>")


if __name__ == "__main__":
    import argparse

    from sweep import pick_tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eval-tasks", type=int, default=48, help="held out: the sweep's tasks are never trained on")
    args = ap.parse_args()
    catalog = json.load(open("lab_tasks.json"))["tasks"]
    held_out = set(pick_tasks(catalog, args.eval_tasks))
    pool = [t for t in pick_tasks(catalog, 10_000) if t not in held_out]
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(
        train, TrainConfig(train_tasks=pool, steps=args.steps))
    print(run.name, run.url)
