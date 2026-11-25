# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""RL on LAB tasks with the loss Harvey describes, on one 8-GPU node, to make checkpoints for the eval matrix.

What matches Harvey's published recipe: rank-64 LoRA over every linear layer, CISPO (the importance weight is
clipped to [0.8, 1.2] and carries no gradient) with Dr. GRPO's advantage (reward minus the group mean, no std) and a
constant loss normalizer (T0 = 55,000 tokens), G = 8 rollouts per task, only model-generated tokens in the loss, and
the reward ``0.35 c + 0.35 h + 0.30 a`` with zero for a cut-off episode.

What doesn't: the adapter covers the attention projections, not the routed experts (see ``Trainer``); it's synchronous (sample with the current adapter, grade, one optimizer step, repeat), so the
importance weight corrects train/inference numerics, not staleness; it's a 20B MoE (3.6B active), not 120B-816B; and the episodes
are capped at a few dozen turns, not 200. The policy is gpt-oss-20b, trained and served as a bf16 copy. This is a way to get checkpoints to measure, not a replication.

    GPUs 0-1   judge (gpt-oss-120b, data-parallel x2)
    GPUs 2-3   policy sampler (vLLM, data-parallel x2, LoRA hot-loaded each step)
    GPUs 4-7   trainer (PyTorch + PEFT, four data-parallel ranks, LoRA gradients all-reduced)

    CLUSTER=amd GPU_QUEUE=<queue> GPU_IMAGE_URI=<uri> python train.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import pickle
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
    attn_implementation: str = "flex_attention"  # gpt-oss's attention sinks rule out SDPA; eager failed on ROCm
    trainer_gpus: int = 4
    max_train_len: int = 32_768  # longer sequences are skipped and counted: cutting context would change the ratios
    # gpt-oss's template drops earlier reasoning, so every turn is its own sequence. Train on this many turns per
    # rollout, sampled at random and weighted by (turns / sampled) so the expected gradient is the full one.
    turns_per_rollout: int = 2
    train_tasks: list[str]
    steps: int = 20
    groups_per_step: int = 8  # N task groups per optimizer step (Harvey: 16)
    group_size: int = 8  # G rollouts per task (Harvey: 8)
    lr: float = 5e-5  # Harvey swept [1e-5, 1e-4] on their smallest model
    lora_rank: int = 64
    lora_alpha: int = 64
    clip_low: float = 0.8  # CISPO importance-weight clip, as published
    clip_high: float = 1.2
    t0: int = 55_000  # Dr. GRPO's constant loss normalizer, as published
    weights: lh.RewardWeights = lh.RewardWeights()
    harness: lh.Harness = lh.Harness(max_turns=30, max_minutes=12)
    temperature: float = 1.0
    max_tokens: int = 8192
    judge_hf: str = "openai/gpt-oss-120b"
    checkpoint_every: int = 5
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
    skipped_too_long: int = 0
    train_tokens: int
    samples: int
    loss: float
    grad_norm: float
    ratio_mean: float
    ratio_clipped: float
    logprobs_matched: float = 0.0  # share of trained tokens with a sampler logprob that lined up
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
        n = len(t.completion_ids)
        ids += extra + t.completion_ids
        mask += [0] * len(extra) + [1] * n
        # NaN marks "no sampler logprob": the trainer uses its own, detached, so that token's ratio is 1.
        lps += [0.0] * len(extra) + (t.logprobs if len(t.logprobs) == n else [float("nan")] * n)
    if ids:
        out.append(Sample(ids=ids, mask=mask, old_logprobs=lps, advantage=advantage))
    return out
# {{/docs-fragment merge}}


BF16_DIR = "/tmp/policy-bf16"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def bf16_copy(hf_id: str, dest: str = BF16_DIR) -> str:
    """gpt-oss ships MXFP4 experts. vLLM's LoRA path for MXFP4 on ROCm builds an emulation kernel at startup that the
    pip ROCm stack can't compile, so write a bf16 copy once (about a minute) and serve and train that."""
    import shutil

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    if os.path.exists(f"{dest}/config.json"):
        return dest
    kw = {"quantization_config": Mxfp4Config(dequantize=True)} if "gpt-oss" in hf_id else {}
    AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, **kw).save_pretrained(dest)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(dest)
    for f in ("chat_template.jinja", "generation_config.json"):
        try:
            shutil.copy(hf_hub_download(hf_id, f), dest)
        except Exception:  # noqa: BLE001  not every repo has both
            pass
    return dest


class Trainer:
    """One data-parallel rank: the policy with a LoRA adapter on one GPU."""

    def __init__(self, cfg: TrainConfig, model_dir: str):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        self.torch, self.cfg = torch, cfg
        # Load the original checkpoint and dequantize in memory, not the bf16 copy vLLM serves: transformers reads the
        # copy's expert weights back as missing and silently re-initializes them (vLLM's loader reads them fine). The
        # weights are the same either way; the diag.py check is that our log-probs match vLLM's scoring of a sequence.
        kw = {}
        if "gpt-oss" in cfg.policy_hf:
            from transformers import Mxfp4Config

            kw["quantization_config"] = Mxfp4Config(dequantize=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.policy_hf, dtype=torch.bfloat16,
                                                     attn_implementation=cfg.attn_implementation, **kw).to("cuda")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        # Harvey's adapter covers the whole network, routed experts included. Ours covers the attention projections:
        # those are the LoRA layers vLLM can hot-load for gpt-oss, and expert LoRA only fits 16k-token sequences here.
        self.model = get_peft_model(model, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                                      target_modules=LORA_TARGETS, lora_dropout=0.0))
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)

    # {{docs-fragment cispo}}
    def backward(self, samples: list[Sample], n_groups: int) -> dict:
        """CISPO: loss = -sum_t sg(clip(r_t, lo, hi)) * A * log pi(t), over model tokens only, divided by
        (groups x G x T0) as in Dr. GRPO, so a long rollout's tokens weigh what a short one's do."""
        torch, cfg = self.torch, self.cfg
        self.model.train()
        denom = n_groups * cfg.group_size * cfg.t0
        ratios, clipped, total_loss, tokens, matched = [], 0, 0.0, 0, 0
        for s in samples:
            ids = torch.tensor([s.ids], device="cuda")
            pos = torch.tensor([i - 1 for i, m in enumerate(s.mask) if m], device="cuda")  # logits that predict them
            tgt = ids[0, pos + 1]
            hidden = self.model.base_model.model.model(input_ids=ids).last_hidden_state[0, pos]
            logp = torch.log_softmax(self.model.base_model.model.lm_head(hidden).float(), -1).gather(-1, tgt[:, None])[:, 0]
            old = torch.tensor([s.old_logprobs[i] for i, m in enumerate(s.mask) if m], device="cuda")
            known = ~torch.isnan(old)
            old = torch.where(known, old, logp.detach())
            ratio = torch.exp(logp.detach() - old)
            w = ratio.clamp(cfg.clip_low, cfg.clip_high)
            loss = -(w * s.advantage * logp).sum() / denom
            loss.backward()
            total_loss += float(loss)
            if known.any():
                ratios.append(float(ratio[known].mean()))
            matched += int(known.sum())
            clipped += int(((ratio < cfg.clip_low) | (ratio > cfg.clip_high)).sum())
            tokens += int(pos.numel())
        return {"loss": total_loss, "ratios": ratios, "clipped": clipped, "train_tokens": tokens, "matched": matched}
    # {{/docs-fragment cispo}}

    def apply(self) -> float:
        """All-reduce the adapter's gradients across ranks (they're small), then one identical step on every rank."""
        torch = self.torch
        import torch.distributed as dist

        if dist.is_initialized():
            for p in self.params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
        gn = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(gn)

    def save(self, path: str) -> str:
        self.model.save_pretrained(path)
        return path

    def load(self, path: str) -> None:
        """Resume: the adapter's weights and the optimizer's state, identical on every rank."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(self.model, load_file(os.path.join(path, "adapter_model.safetensors")))
        self.opt.load_state_dict(self.torch.load(os.path.join(path, "optimizer.pt"), map_location="cuda"))


def _rank_main(rank: int, world: int, gpu: int, cfg_json: str, model_dir: str, inq, outq) -> None:
    os.environ[servers.VISIBLE] = str(gpu)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    import torch.distributed as dist

    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world)  # RCCL on ROCm
    trainer = Trainer(TrainConfig.model_validate_json(cfg_json), model_dir)
    outq.put(("ready", rank))
    while True:
        cmd, arg = inq.get()
        if cmd == "step":
            samples, n_groups = arg
            stats = trainer.backward(samples, n_groups)
            stats["grad_norm"] = trainer.apply()
            outq.put(("stats", stats))
        elif cmd == "save":
            if rank == 0:
                trainer.save(arg)
                import torch

                torch.save(trainer.opt.state_dict(), os.path.join(arg, "optimizer.pt"))
            outq.put(("saved", arg))
        elif cmd == "load":
            trainer.load(arg)
            outq.put(("loaded", arg))
        elif cmd == "stop":
            return


class TrainerPool:
    """Data-parallel ranks in their own processes, so each one sees only its GPU. Samples are dealt out by length."""

    def __init__(self, cfg: TrainConfig, model_dir: str, gpus: list[int]):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.inqs = [ctx.Queue() for _ in gpus]
        self.outq = ctx.Queue()
        self.procs = [ctx.Process(target=_rank_main, args=(r, len(gpus), g, cfg.model_dump_json(), model_dir,
                                                           self.inqs[r], self.outq), daemon=True)
                      for r, g in enumerate(gpus)]
        for p in self.procs:
            p.start()
        for _ in gpus:
            self._get("ready")

    def _get(self, kind: str, timeout: float = 7200):
        msg = self.outq.get(timeout=timeout)
        if msg[0] != kind:
            raise RuntimeError(f"expected {kind}, got {msg[0]}")
        return msg[1]

    def step(self, samples: list[Sample], n_groups: int, clip_low: float, clip_high: float) -> dict:
        shards: list[list[Sample]] = [[] for _ in self.inqs]
        loads = [0] * len(self.inqs)
        for s in sorted(samples, key=lambda s: -len(s.ids)):  # longest first onto the least-loaded rank
            i = loads.index(min(loads))
            shards[i].append(s)
            loads[i] += len(s.ids)
        for q, shard in zip(self.inqs, shards):
            q.put(("step", (shard, n_groups)))
        parts = [self._get("stats") for _ in self.inqs]
        ratios = [r for p in parts for r in p["ratios"]]
        tokens = sum(p["train_tokens"] for p in parts)
        return {"loss": sum(p["loss"] for p in parts), "grad_norm": parts[0]["grad_norm"],
                "ratio_mean": st.fmean(ratios) if ratios else 1.0,
                "ratio_clipped": sum(p["clipped"] for p in parts) / max(tokens, 1), "train_tokens": tokens,
                "logprobs_matched": sum(p["matched"] for p in parts) / max(tokens, 1)}

    def save(self, path: str) -> str:
        for q in self.inqs:
            q.put(("save", path))
        for _ in self.inqs:
            self._get("saved")
        return path

    def load(self, path: str) -> None:
        for q in self.inqs:
            q.put(("load", path))
        for _ in self.inqs:
            self._get("loaded")

    def stop(self) -> None:
        for q in self.inqs:
            q.put(("stop", None))


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


def _resume_prefix() -> str | None:
    """A path that's the same on every attempt of this action, so a retry on a new node can pick up where the
    reclaimed one stopped. Attempt directories look like .../<run>/<action>/<run>-<action>-<attempt>/..."""
    try:
        from flyte._context import internal_ctx

        raw = internal_ctx().raw_data.path
        tctx = flyte.ctx()
        key = f"/{tctx.action.run_name}/{tctx.action.name}/"
        return raw[: raw.index(key) + len(key)] + "resume/" if key in raw else None
    except Exception:  # noqa: BLE001  local runs: no resume
        return None


async def _save_resume(prefix: str | None, local_adapter: str, state: dict) -> None:
    if not prefix:
        return
    import flyte.storage

    Path(local_adapter, "state.json").write_text(json.dumps(state))
    await flyte.storage.put(local_adapter, prefix + "latest", recursive=True)


async def _load_resume(prefix: str | None) -> tuple[str, dict] | None:
    if not prefix:
        return None
    import flyte.storage

    try:
        local = await flyte.storage.get(prefix + "latest", "/tmp/resume", recursive=True)
    except Exception:  # noqa: BLE001  nothing saved yet
        return None
    p = next(Path("/tmp/resume").rglob("state.json"), None)
    return (str(p.parent), json.loads(p.read_text())) if p else None


@flyte.trace
async def traced_groups(lab_root: str, task_id: str, model: str, base_url: str) -> dict[str, list[str]]:
    import openai

    task, _ = lh.load_task(Path(lab_root), task_id)
    return await asyncio.to_thread(lh.group_issues, task, openai.OpenAI(base_url=base_url, api_key="x"), model)


# {{docs-fragment train}}
@train_env.task(cache="auto", retries=2, report=True)
async def train(cfg: TrainConfig) -> TrainResult:
    t_start = time.time()
    lab = lh.fetch_lab(cfg.train_tasks, LAB_COMMIT, Path("/tmp/lab"))
    amd = CLUSTER == "amd"
    judge_srv = servers.ServerSpec(served_name="judge", hf_id=cfg.judge_hf, gpus=[0, 1] if amd else [0],
                                   max_model_len=131072, gpu_memory_utilization=0.85 if amd else 0.3)
    model_dir = await asyncio.to_thread(bf16_copy, cfg.policy_hf)
    tool, reasoning = servers.parsers_for(cfg.policy_hf)
    pol_srv = servers.ServerSpec(served_name="policy", hf_id=model_dir, gpus=[2, 3] if amd else [0],
                                 tool_parser=tool, reasoning_parser=reasoning, max_model_len=131072, enable_lora=True, max_lora_rank=cfg.lora_rank,
                                 gpu_memory_utilization=0.85 if amd else 0.3, port=8300)
    await asyncio.gather(servers.ensure(judge_srv), servers.ensure(pol_srv))
    judge = lh.OpenAICompatJudge(lh.JudgeSpec(name="judge", hf_id=cfg.judge_hf, reasoning_effort="low"), judge_srv.url)
    groups = dict(zip(cfg.train_tasks, await asyncio.gather(*[
        traced_groups(str(lab), t, "judge", judge_srv.url) for t in cfg.train_tasks])))

    trainer = TrainerPool(cfg, model_dir, list(range(8 - cfg.trainer_gpus, 8)) if amd else [0])
    rng = random.Random(cfg.seed)
    history: list[StepLog] = []
    ckpts: dict[int, flyte.io.Dir] = {}
    adapter, first = "policy", 0  # step 0 samples from the base model
    prefix = _resume_prefix()
    print(f"resume prefix: {prefix}", flush=True)
    resumed = await _load_resume(prefix)
    if resumed:  # a retry after the node was taken back: continue from the last finished step
        path, state = resumed
        trainer.load(path)
        first, adapter = state["next_step"], state["adapter"]
        history = [StepLog(**h) for h in state["history"]]
        ckpts = {int(k): flyte.io.Dir.from_existing_remote(v) for k, v in state["checkpoints"].items()}
        rng.setstate(pickle.loads(base64.b64decode(state["rng"])))
        await servers.load_lora(pol_srv, adapter, path)
        print(f"resumed at step {first} from {prefix}", flush=True)
    for step in range(first, cfg.steps + 1):
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
        too_long = sum(len(s.ids) > cfg.max_train_len for s in samples)
        samples = [s for s in samples if len(s.ids) <= cfg.max_train_len]
        stats = trainer.step(samples, cfg.groups_per_step, cfg.clip_low, cfg.clip_high) if samples else {
            "loss": 0.0, "grad_norm": 0.0, "ratio_mean": 1.0, "ratio_clipped": 0.0, "train_tokens": 0, "logprobs_matched": 0.0}
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
                      groups_with_signal=live, samples=len(samples), skipped_too_long=too_long, rollout_s=t1 - t0, grade_s=t2 - t1,
                      train_s=t3 - t2, **stats)
        history.append(log)
        print(log.model_dump_json(), flush=True)
        if (step + 1) % cfg.checkpoint_every == 0:
            ckpts[step + 1] = await flyte.io.Dir.from_local(path)
        await _save_resume(prefix, path, {
            "next_step": step + 1, "adapter": adapter, "history": [h.model_dump() for h in history],
            "checkpoints": {k: d.path for k, d in ckpts.items()},
            "rng": base64.b64encode(pickle.dumps(rng.getstate())).decode()})
        await flyte.report.replace.aio(_report(history, time.time() - t_start), do_flush=True)
    trainer.stop()
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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eval-tasks", type=int, default=48, help="held out: the sweep's tasks are never trained on")
    args = ap.parse_args()
    catalog = json.load(open("lab_tasks.json"))["tasks"]
    held_out = set(pick_tasks(catalog, args.eval_tasks))
    pool = [t for t in pick_tasks(catalog, 10_000) if t not in held_out]
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(
        train, TrainConfig(train_tasks=pool, steps=args.steps))
    print(run.name, run.url)
