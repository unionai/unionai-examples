# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.0",
#    "torch>=2.5.0",
#    "transformers>=4.51.0",
#    "peft>=0.13.0",
#    "vllm>=0.11.0",
#    "pandas", "pyarrow",
# ]
# main = "screen"
# params = ""
# ///
"""Can a LoRA screen pick the right RL ablation? Run the same arms at LoRA and at full weight, and compare.

Mercor's 397B guide ran 13 full-weight ablations on a 35B model to pick a loss aggregation (``prompt_mean``
won by +3.9 points). This example asks whether a cheaper screen would have picked the same winner: every
loss-aggregation arm is trained twice from the same base, once as a LoRA adapter and once at full weight,
and the screen reports whether the two rankings agree.

What's real here and what isn't: a 0.6B model on GSM8K, short single-turn completions, a single seed by
default. It tests the *method* on a cheap proxy. It says nothing yet about long-horizon agentic RL, where
the trajectory-length spread that ``prompt_mean`` exists for is 2k-128k tokens, not 50-500.

Structure:

- ``prefetch``/``load_gsm8k``  base weights and data, cached once.
- ``train_arm``   one arm = one durable action on one GPU: sample with vLLM (subprocess, token-in/token-out),
                  score, one GRPO step with the arm's loss aggregation, checkpoint, repeat; then a greedy
                  held-out eval. LoRA vs full weight is a field on the arm, not a different program.
- ``screen``      fans out the arms, then reports held-out accuracy per arm and whether LoRA and full weight
                  rank the aggregations the same way. Add an arm and only that arm runs; the rest are cached.

Run::

    python lora_screen.py              # 3 aggregations × {lora, full} × 2 seeds = 12 arms, one L4 each
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from datetime import timedelta
from itertools import combinations
from pathlib import Path

import flyte
import flyte.io
import flyte.report
import vllm_generate  # the sampler script; imported so it ships in the code bundle
from pydantic import BaseModel

GPU_TYPE = os.environ.get("SCREEN_GPU_TYPE", "L4")

# {{docs-fragment image}}
# Same recipe as the rl_grpo_lora tutorial: prebuilt flashinfer kernels so vLLM never needs nvcc.
image = (
    flyte.Image.from_debian_base(name="lora-ablation-screen", python_version=(3, 12))
    .with_pip_packages("flashinfer-python", "flashinfer-cubin")
    .with_pip_packages("flashinfer-jit-cache", index_url="https://flashinfer.ai/whl/cu129")
    .with_pip_packages("vllm==0.11.0", "transformers==4.57.6", "peft>=0.13.0", "accelerate>=0.34.0",
                       "pandas", "pyarrow")
)
# {{/docs-fragment image}}
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")

data_env = flyte.TaskEnvironment(name="lora-screen-data", image=image,
                                 resources=flyte.Resources(cpu=2, memory="4Gi"), secrets=[HF_SECRET])
# {{docs-fragment arm_env}}
arm_env = flyte.TaskEnvironment(
    name="lora-screen-arm",
    image=image,
    resources=flyte.Resources(cpu=8, memory="48Gi", gpu=flyte.GPU(GPU_TYPE, 1), disk="60Gi", shm="auto"),
    env_vars={"VLLM_USE_V1": "1", "TOKENIZERS_PARALLELISM": "false"},
)
# {{/docs-fragment arm_env}}
driver_env = flyte.TaskEnvironment(name="lora-screen-driver", image=image,
                                   resources=flyte.Resources(cpu=1, memory="2Gi"), depends_on=[data_env, arm_env])


# --- config -------------------------------------------------------------------------------------------


# {{docs-fragment arms}}
class Arm(BaseModel):
    """One row of the ablation table. Everything that defines the arm is here, so it's the cache key."""

    loss_agg: str  # token_mean | sequence_mean | prompt_mean
    mode: str  # lora | full
    lr: float
    seed: int = 0

    @property
    def name(self) -> str:
        return f"{self.mode}-{self.loss_agg}-s{self.seed}"


class ScreenConfig(BaseModel):
    base_model: str = "Qwen/Qwen3-0.6B"
    steps: int = 20
    prompts_per_step: int = 16
    group_size: int = 8
    max_tokens: int = 512
    eval_questions: int = 300
    lora_rank: int = 16
    lora_alpha: int = 32
    # Without these the model writes "#### 42" and keeps going until max_tokens, every completion has the same
    # length, and the three aggregations become the same loss (an accidental A/A test; see the README).
    stop: list[str] = ["\nProblem:", "Problem:", "\n\n\n"]


# LoRA wants roughly 10x the full-weight learning rate (Thinking Machines, "LoRA Without Regret").
DEFAULT_ARMS = [Arm(loss_agg=agg, mode=mode, lr=2e-5 if mode == "lora" else 2e-6, seed=seed)
                for agg in ("token_mean", "sequence_mean", "prompt_mean") for mode in ("lora", "full")
                for seed in (0, 1)]
# {{/docs-fragment arms}}


class QA(BaseModel):
    question: str
    answer: str


class StepLog(BaseModel):
    step: int
    mean_reward: float
    mean_len: float
    len_cv: float  # coefficient of variation of completion length: what aggregation choices are sensitive to
    engine_init_s: float
    generate_s: float
    train_s: float


class ArmResult(BaseModel):
    arm: Arm
    eval_acc: float
    eval_se: float
    history: list[StepLog]
    seconds_total: float
    policy_bytes: int  # what moves trainer -> sampler each step: an adapter or a whole model


# --- data ---------------------------------------------------------------------------------------------


PROMPT = ("Solve the problem. Reason briefly, then give the final integer answer on its own line as '#### <answer>'.\n\n"
          "Problem: {q}\nSolution:")


def _final_number(text: str) -> str | None:
    tail = text.rsplit("####", 1)[1] if "####" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[0 if "####" in text else -1].replace(",", "").rstrip(".") if nums else None


def reward(completion: str, answer: str) -> float:
    got = _final_number(completion)
    try:
        return 1.0 if got is not None and abs(float(got) - float(answer)) < 1e-6 else 0.0
    except ValueError:
        return 0.0


@data_env.task(cache="auto")
async def load_gsm8k(n_train: int = 2000, n_test: int = 300) -> tuple[list[QA], list[QA]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def load(split: str, n: int) -> list[QA]:
        path = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path).head(n)
        return [QA(question=q, answer=a.split("####")[-1].strip().replace(",", "")) for q, a in zip(df.question, df.answer)]

    return load("train", n_train), load("test", n_test)


@data_env.task(cache="auto")
async def prefetch(base_model: str) -> flyte.io.Dir:
    from huggingface_hub import snapshot_download

    local = await asyncio.to_thread(snapshot_download, base_model, allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                                                  "*.model", "tokenizer*"])
    return await flyte.io.Dir.from_local(local)


# --- the GRPO step: where the arms differ -------------------------------------------------------------


# {{docs-fragment aggregation}}
def sequence_weights(lengths: list[int], groups: list[int], loss_agg: str) -> list[float]:
    """Per-sequence weight w_i such that loss = -sum_i A_i * w_i * sum_t log p(t).

    token_mean     one pool of tokens: long completions dominate the gradient
    sequence_mean  every completion counts equally, whatever its length (original GRPO)
    prompt_mean    every prompt group counts equally; tokens are pooled within the group (DAPO-style)
    """
    n_seq, n_tok = len(lengths), sum(lengths)
    if loss_agg == "token_mean":
        return [1.0 / n_tok] * n_seq
    if loss_agg == "sequence_mean":
        return [1.0 / (n_seq * max(L, 1)) for L in lengths]
    if loss_agg == "prompt_mean":
        group_tok: dict[int, int] = {}
        for g, L in zip(groups, lengths):
            group_tok[g] = group_tok.get(g, 0) + L
        n_groups = len(group_tok)
        return [1.0 / (n_groups * max(group_tok[g], 1)) for g in groups]
    raise ValueError(loss_agg)
# {{/docs-fragment aggregation}}


def advantages(rewards: list[float], groups: list[int]) -> list[float]:
    by: dict[int, list[float]] = {}
    for g, r in zip(groups, rewards):
        by.setdefault(g, []).append(r)
    stats = {g: (statistics.fmean(v), statistics.pstdev(v)) for g, v in by.items()}
    return [(r - stats[g][0]) / (stats[g][1] + 1e-4) for g, r in zip(groups, rewards)]


async def _sample(policy_dir: str, lora_dir: str | None, prompts: list[str], n: int, cfg: ScreenConfig,
                  temperature: float, seed: int, workdir: Path) -> dict:
    req, out = workdir / "req.json", workdir / "out.json"
    req.write_text(json.dumps(prompts))
    cmd = [sys.executable, vllm_generate.__file__, "--model", policy_dir,
           "--requests", str(req), "--out", str(out), "--n", str(n), "--max-tokens", str(cfg.max_tokens),
           "--temperature", str(temperature), "--seed", str(seed), "--gpu-mem", "0.35",
           "--lora-rank", str(cfg.lora_rank), "--stop", json.dumps(cfg.stop)]
    if lora_dir:
        cmd += ["--lora", lora_dir]
    proc = await asyncio.create_subprocess_exec(*cmd)
    if await proc.wait() != 0:
        raise RuntimeError(f"vLLM sampling failed with exit code {proc.returncode}")
    return json.loads(out.read_text())


# {{docs-fragment train_arm}}
@arm_env.task(cache="auto", retries=2, timeout=flyte.Timeout(max_runtime=timedelta(hours=3)))
async def train_arm(base: flyte.io.Dir, arm: Arm, cfg: ScreenConfig, train: list[QA], test: list[QA]) -> ArmResult:
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.monotonic()
    torch.manual_seed(arm.seed)
    local_base = await base.download()
    tok = AutoTokenizer.from_pretrained(local_base)
    # fp32 master weights for full-weight RL (bf16 can't represent lr=2e-6 updates); bf16 base under LoRA.
    model = AutoModelForCausalLM.from_pretrained(
        local_base, torch_dtype=torch.float32 if arm.mode == "full" else torch.bfloat16).cuda()
    if arm.mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=arm.lr, betas=(0.9, 0.98),
                            weight_decay=0.0)
    work = Path(tempfile.mkdtemp(prefix=f"{arm.name}-"))
    policy_dir = work / "policy"

    def export_policy() -> tuple[str, str | None]:
        """Write the current policy where the sampler can load it: an adapter, or a whole model."""
        model.save_pretrained(policy_dir)
        if arm.mode == "lora":
            return local_base, str(policy_dir)
        tok.save_pretrained(policy_dir)
        return str(policy_dir), None

    rng = random.Random(arm.seed)
    history: list[StepLog] = []
    for step in range(cfg.steps):
        batch = rng.sample(train, cfg.prompts_per_step)
        sampler_model, lora = export_policy()
        out = await _sample(sampler_model, lora, [PROMPT.format(q=b.question) for b in batch], cfg.group_size, cfg,
                            1.0, arm.seed * 1000 + step, work)
        seqs, groups, rewards = [], [], []
        for g, (qa, grp) in enumerate(zip(batch, out["groups"])):
            for s in grp["samples"]:
                seqs.append((grp["prompt_ids"], s["ids"]))
                groups.append(g)
                rewards.append(reward(s["text"], qa.answer))
        adv = advantages(rewards, groups)
        lengths = [len(c) for _, c in seqs]
        weights = sequence_weights(lengths, groups, arm.loss_agg)

        t0 = time.monotonic()
        model.train()
        opt.zero_grad(set_to_none=True)
        for (p_ids, c_ids), a, w in zip(seqs, adv, weights):
            if a == 0.0 or not c_ids:
                continue  # no signal: the whole group scored the same
            ids = torch.tensor([p_ids + c_ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids).logits[:, len(p_ids) - 1 : -1, :].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[:, len(p_ids):].unsqueeze(-1)).squeeze(-1)
            (-a * w * logp.sum()).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        history.append(StepLog(
            step=step, mean_reward=round(statistics.fmean(rewards), 4), mean_len=round(statistics.fmean(lengths), 1),
            len_cv=round(statistics.pstdev(lengths) / max(statistics.fmean(lengths), 1), 3),
            engine_init_s=out["engine_init_s"], generate_s=out["generate_s"], train_s=round(time.monotonic() - t0, 1)))
        print(json.dumps(history[-1].model_dump()), flush=True)

    # held-out eval: greedy, one sample per question
    sampler_model, lora = export_policy()
    ev = await _sample(sampler_model, lora, [PROMPT.format(q=q.question) for q in test], 1, cfg, 0.0, 0, work)
    correct = [reward(g["samples"][0]["text"], qa.answer) for qa, g in zip(test, ev["groups"])]
    acc = statistics.fmean(correct)
    policy_bytes = sum(f.stat().st_size for f in policy_dir.rglob("*.safetensors"))
    return ArmResult(arm=arm, eval_acc=round(acc, 4), eval_se=round(math.sqrt(acc * (1 - acc) / len(correct)), 4),
                     history=history, seconds_total=round(time.monotonic() - t_start, 1), policy_bytes=policy_bytes)
# {{/docs-fragment train_arm}}


@arm_env.task(cache="auto")
async def eval_base(base: flyte.io.Dir, cfg: ScreenConfig, test: list[QA]) -> float:
    local_base = await base.download()
    work = Path(tempfile.mkdtemp(prefix="base-"))
    ev = await _sample(local_base, None, [PROMPT.format(q=q.question) for q in test], 1, cfg, 0.0, 0, work)
    return round(statistics.fmean(reward(g["samples"][0]["text"], qa.answer) for qa, g in zip(test, ev["groups"])), 4)


# --- the screen ----------------------------------------------------------------------------------------


class ScreenSummary(BaseModel):
    base_acc: float
    results: list[ArmResult]
    ranking: dict[str, list[str]]  # mode -> loss aggregations, best first
    kendall_tau: float
    same_winner: bool


def kendall_tau(a: list[str], b: list[str]) -> float:
    pos = {x: i for i, x in enumerate(b)}
    pairs = list(combinations(a, 2))
    s = sum(1 if pos[x] < pos[y] else -1 for x, y in pairs)
    return s / len(pairs) if pairs else 1.0


def _report(s: ScreenSummary) -> str:
    rows = "".join(
        f"<tr><td>{r.arm.name}</td><td>{r.arm.lr:g}</td><td>{r.eval_acc:.1%} ± {r.eval_se:.1%}</td>"
        f"<td>{r.history[0].mean_reward:.2f} → {r.history[-1].mean_reward:.2f}</td>"
        f"<td>{statistics.fmean(h.len_cv for h in r.history):.2f}</td>"
        f"<td>{statistics.fmean(h.engine_init_s for h in r.history):.0f} / "
        f"{statistics.fmean(h.generate_s for h in r.history):.0f} / {statistics.fmean(h.train_s for h in r.history):.0f}</td>"
        f"<td>{r.policy_bytes / 1e6:,.0f} MB</td><td>{r.seconds_total / 60:.0f}</td></tr>"
        for r in sorted(s.results, key=lambda r: (r.arm.mode, -r.eval_acc)))
    return (f"<h2>LoRA screen vs full weight</h2><p>base model held-out accuracy: {s.base_acc:.1%}</p>"
            "<table border=1 cellpadding=4><tr><th>arm</th><th>lr</th><th>held-out acc</th><th>train reward first → last</th>"
            "<th>length CV</th><th>engine init / generate / train (s per step)</th><th>policy per step</th>"
            f"<th>minutes</th></tr>{rows}</table><p>ranking: {s.ranking}</p>"
            f"<p>Kendall τ (LoRA vs full): {s.kendall_tau:+.2f} · same winner: {s.same_winner}</p>")


# {{docs-fragment screen}}
@driver_env.task(report=True)
async def screen(arms: list[Arm] = DEFAULT_ARMS, cfg: ScreenConfig = ScreenConfig()) -> ScreenSummary:
    (train, test), base = await asyncio.gather(load_gsm8k(n_test=cfg.eval_questions), prefetch(cfg.base_model))
    base_acc_task = asyncio.create_task(eval_base(base, cfg, test))
    outs = await asyncio.gather(*(train_arm(base, a, cfg, train, test) for a in arms), return_exceptions=True)
    results = [o for o in outs if isinstance(o, ArmResult)]  # a failed arm is a gap in the table, not a dead screen
    def mean_acc(mode: str, agg: str) -> float:
        accs = [r.eval_acc for r in results if r.arm.mode == mode and r.arm.loss_agg == agg]
        return statistics.fmean(accs) if accs else float("nan")

    aggs = sorted({r.arm.loss_agg for r in results})
    ranking = {mode: sorted((a for a in aggs if not math.isnan(mean_acc(mode, a))), key=lambda a: -mean_acc(mode, a))
               for mode in ("lora", "full")}  # mean held-out accuracy over seeds, best first
    common = [x for x in ranking["full"] if x in ranking["lora"]]
    lora_order = [x for x in ranking["lora"] if x in common]
    summary = ScreenSummary(
        base_acc=await base_acc_task, results=results, ranking=ranking,
        kendall_tau=round(kendall_tau(lora_order, common), 3),
        same_winner=bool(common) and lora_order[0] == common[0],
    )
    await flyte.report.replace.aio(_report(summary), do_flush=True)
    return summary
# {{/docs-fragment screen}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(screen).url)
