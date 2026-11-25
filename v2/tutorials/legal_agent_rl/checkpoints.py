# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Choosing a checkpoint: a matrix of (checkpoint x suite) cells and a pairwise tournament, each cell cached.

Harvey evaluated hold-out sets every 20 steps and picked step 80 for four models and step 200 for one. That choice
is an eval matrix: checkpoints down one side, suites across the top. Here every cell is its own cached task in a
*reusable* environment, so the cells share one warm 8-GPU node (the policy server holds every checkpoint as a LoRA
adapter, the judge stays loaded), and asking for a new checkpoint or a new suite runs only the new cells.

Suites:
    lab       held-out LAB tasks (never trained on): reward terms, turns, length, truncation
    cuad      CUAD clause extraction (never trained on): token F1, the benchmark Harvey reports regressing
    pairs     every pair of checkpoints on the lab suite's deliverables, both orders -> Bradley-Terry / Elo

    CLUSTER=amd GPU_QUEUE=<queue> python checkpoints.py --train-run <run name>
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import statistics as st
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import flyte
import flyte.io
import flyte.report
from pydantic import BaseModel, Field

import lab_harness as lh
import preference as pref
import servers
from config import PROPAGATE, cpu_env_kwargs, gpu_env_kwargs
from images import LAB_COMMIT

# {{docs-fragment reusable}}
cell_env = flyte.TaskEnvironment(
    name="legal-rl-cells", env_vars=PROPAGATE,
    # One warm node for every cell: servers started by the first cell stay up for the rest.
    reusable=flyte.ReusePolicy(replicas=1, concurrency=12, idle_ttl=1800, scaledown_ttl=1800),
    **gpu_env_kwargs())
# {{/docs-fragment reusable}}
matrix_env = flyte.TaskEnvironment(name="legal-rl-matrix", env_vars=PROPAGATE, depends_on=[cell_env],
                                   **cpu_env_kwargs())

from train import BF16_DIR, bf16_copy  # noqa: E402

POLICY_HF = "openai/gpt-oss-20b"
POLICY = servers.ServerSpec(served_name="base", hf_id=BF16_DIR, gpus=[2, 3], tool_parser="openai",
                            reasoning_parser="openai_gptoss", max_model_len=131072, enable_lora=True, max_loras=8,
                            port=8300)
# The judge the policy was trained against, and a judge from another family it never saw.
JUDGE = servers.ServerSpec(served_name="gpt-oss-120b", hf_id="openai/gpt-oss-120b", gpus=[0, 1], max_model_len=131072)
JUDGE_B = servers.ServerSpec(served_name="nemotron-3-nano-30b", hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                             gpus=[4, 5, 6, 7], max_model_len=131072)
_loaded: set[str] = set()
_lock = asyncio.Lock()


class Checkpoint(BaseModel):
    step: int
    adapter: flyte.io.Dir | None = None  # None: the base model

    @property
    def name(self) -> str:
        return "base" if self.adapter is None else f"step-{self.step:04d}"


class Suite(BaseModel):
    kind: str  # lab | cuad
    task_ids: list[str] = []
    samples: int = 4
    cuad_items: int = 120
    harness: lh.Harness = lh.Harness(max_turns=30)


class CellResult(BaseModel):
    checkpoint: str
    step: int
    suite: str
    metrics: dict
    rollouts: list[lh.Rollout] = Field(default_factory=list)  # lab suite: kept for the tournament


async def _ready(ck: Checkpoint) -> str:
    async with _lock:
        await asyncio.to_thread(bf16_copy, POLICY_HF)
    await asyncio.gather(servers.ensure(POLICY), servers.ensure(JUDGE), servers.ensure(JUDGE_B))
    async with _lock:
        if ck.adapter is not None and ck.name not in _loaded:
            local = await ck.adapter.download()
            await servers.load_lora(POLICY, ck.name, local)
            _loaded.add(ck.name)
    return ck.name


# --- the lab suite --------------------------------------------------------------------------------------------


@cell_env.task(cache="auto", retries=3)
async def lab_cell(ck: Checkpoint, suite: Suite) -> CellResult:
    model = await _ready(ck)
    lab = lh.fetch_lab(suite.task_ids, LAB_COMMIT, Path("/tmp/lab"))
    policy = lh.Policy(name=model, hf_id=POLICY_HF, temperature=1.0, max_tokens=8192, reasoning_effort="medium")
    sem = asyncio.Semaphore(48)

    async def one(t, k):
        async with sem:
            return await asyncio.to_thread(lh.run_rollout, lab, t, policy, suite.harness, POLICY.url, k)

    rollouts = await asyncio.gather(*[one(t, k) for t in suite.task_ids for k in range(suite.samples)])
    judge = lh.OpenAICompatJudge(lh.JudgeSpec(name=JUDGE.served_name, hf_id=JUDGE.hf_id, reasoning_effort="low"),
                                 JUDGE.url)
    judge_b = lh.OpenAICompatJudge(lh.JudgeSpec(name=JUDGE_B.served_name, hf_id=JUDGE_B.hf_id, thinking=False),
                                   JUDGE_B.url)
    gsem = asyncio.Semaphore(32)

    async def grade_both(r):
        async with gsem:
            task, _ = lh.load_task(lab, r.task_id)
            va, vb = await asyncio.gather(asyncio.to_thread(lh.grade, r, task, judge, JUDGE.served_name, 16),
                                          asyncio.to_thread(lh.grade, r, task, judge_b, JUDGE_B.served_name, 16))
            return lh.reward_terms(va.verdicts, {}), lh.reward_terms(vb.verdicts, {})

    both = await asyncio.gather(*[grade_both(r) for r in rollouts])
    terms, terms_b = [x[0] for x in both], [x[1] for x in both]
    rewards = [lh.reward(x, r.finish_reason) for x, r in zip(terms, rollouts)]
    m = {"reward": st.fmean(rewards), "criteria_pass": st.fmean(x["c"] for x in terms),
         "criteria_pass_judge_b": st.fmean(x["c"] for x in terms_b),
         "all_pass": st.fmean(x["a"] for x in terms),
         "truncated": st.fmean(r.finish_reason in lh.TRUNCATED for r in rollouts),
         "no_deliverable": st.fmean(r.deliverable_chars == 0 for r in rollouts),
         "turns": st.fmean(r.turns for r in rollouts), "output_tokens": st.fmean(r.output_tokens for r in rollouts),
         "deliverable_bytes": st.fmean(r.deliverable_chars for r in rollouts),
         "finish": dict(Counter(r.finish_reason for r in rollouts))}
    return CellResult(checkpoint=ck.name, step=ck.step, suite="lab", metrics=m, rollouts=rollouts)


# --- the CUAD suite -------------------------------------------------------------------------------------------

CUAD_PROMPT = """Below is a commercial contract, then a question about one kind of clause.

## Contract
{context}

## Question
{question}

Quote the exact sentence or sentences from the contract that answer the question. If the contract has no such
clause, answer exactly: None"""


def _tokens(s: str) -> list[str]:
    s = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", s.lower()))
    return s.split()


def token_f1(pred: str, golds: list[str]) -> float:
    """SQuAD-style token F1, the max over gold answers; an empty gold list means the right answer is 'None'."""
    p = pred.strip()
    if not golds:
        return 1.0 if p.lower().startswith("none") else 0.0
    if p.lower().startswith("none"):
        return 0.0
    best = 0.0
    for g in golds:
        pt, gt = _tokens(p), _tokens(g)
        common = sum((Counter(pt) & Counter(gt)).values())
        if common:
            pr, rc = common / len(pt), common / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def cuad_items(n: int, seed: int = 0, max_chars: int = 120_000) -> list[dict]:
    """A fixed sample of CUAD questions (CC BY 4.0, The Atticus Project), about 1 in 4 with no answer."""
    from huggingface_hub import hf_hub_download

    data = json.load(open(hf_hub_download("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json", repo_type="dataset")))
    rng = random.Random(seed)
    qas = [{"id": qa["id"], "context": para["context"], "question": qa["question"],
            "answers": sorted({a["text"] for a in qa["answers"]})}
           for doc in data["data"] for para in doc["paragraphs"] if len(para["context"]) <= max_chars
           for qa in para["qas"]]
    has, none = [q for q in qas if q["answers"]], [q for q in qas if not q["answers"]]
    k_none = n // 4
    return sorted(rng.sample(has, n - k_none) + rng.sample(none, k_none), key=lambda q: q["id"])


@cell_env.task(cache="auto", retries=3)
async def cuad_cell(ck: Checkpoint, suite: Suite) -> CellResult:
    import openai

    model = await _ready(ck)
    items = await asyncio.to_thread(cuad_items, suite.cuad_items)
    client = openai.AsyncOpenAI(base_url=POLICY.url, api_key="x", timeout=900)
    sem = asyncio.Semaphore(48)

    async def one(q):
        async with sem:
            r = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=1024, messages=[
                {"role": "user", "content": CUAD_PROMPT.format(context=q["context"], question=q["question"])}])
            text = r.choices[0].message.content or ""
            return token_f1(text, q["answers"]), len(text), r.choices[0].finish_reason

    res = await asyncio.gather(*[one(q) for q in items])
    f1 = [x[0] for x in res]
    has = [x for x, q in zip(f1, items) if q["answers"]]
    m = {"f1": st.fmean(f1), "f1_has_answer": st.fmean(has), "none_correct": st.fmean(
        x for x, q in zip(f1, items) if not q["answers"]), "answer_chars": st.fmean(x[1] for x in res),
         "cut_off": st.fmean(x[2] == "length" for x in res), "items": len(items)}
    return CellResult(checkpoint=ck.name, step=ck.step, suite="cuad", metrics=m)


# --- the tournament -------------------------------------------------------------------------------------------


def _deliverable_md(r: lh.Rollout) -> str:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = lh.untar_b64(r.outputs_b64, Path(d))
        docx = next(out.rglob("*.docx"), None)
        if docx is None:
            return "(no deliverable)"
        return subprocess.run(["pandoc", str(docx), "-t", "gfm"], capture_output=True, text=True).stdout


# {{docs-fragment pair_cell}}
@cell_env.task(cache="auto", retries=3)
async def pair_cell(a: CellResult, b: CellResult, task_ids: list[str]) -> list[tuple[str, str, float]]:
    """Every task, sample-matched: a's k-th deliverable against b's k-th, in both orders. a == b is the A/A test."""
    import openai

    await servers.ensure(JUDGE)
    lab = lh.fetch_lab(task_ids, LAB_COMMIT, Path("/tmp/lab"))
    client = openai.OpenAI(base_url=JUDGE.url, api_key="x", timeout=900)
    by = lambda c: {(r.task_id, r.sample): r for r in c.rollouts}  # noqa: E731
    ra, rb = by(a), by(b)
    keys = sorted(k for k in ra if k in rb)
    if a.checkpoint == b.checkpoint:  # A/A: pair sample k with sample k+1 of the same checkpoint
        keys = [(t, k) for t, k in keys if (t, k + 1) in rb]
    sem = asyncio.Semaphore(32)

    async def one(t, k):
        async with sem:
            task, _ = lh.load_task(lab, t)
            other = rb[(t, k + 1)] if a.checkpoint == b.checkpoint else rb[(t, k)]
            res = await asyncio.to_thread(pref.both_orders, client, JUDGE.served_name, task.instructions,
                                          _deliverable_md(ra[(t, k)]), _deliverable_md(other))
            return (a.checkpoint, b.checkpoint + ("'" if a.checkpoint == b.checkpoint else ""), res["score"])

    return await asyncio.gather(*[one(t, k) for t, k in keys])
# {{/docs-fragment pair_cell}}


# --- the driver -----------------------------------------------------------------------------------------------


class Matrix(BaseModel):
    cells: list[CellResult]
    elo: dict
    win_rates: dict
    table: list[dict]


@matrix_env.task(report=True)
async def matrix(checkpoints: list[Checkpoint], lab_suite: Suite, cuad_suite: Suite) -> Matrix:
    t0 = time.time()
    lab_cells = await asyncio.gather(*[lab_cell(ck, lab_suite) for ck in checkpoints])
    cuad_cells = await asyncio.gather(*[cuad_cell(ck, cuad_suite) for ck in checkpoints])
    pairs = [(x, y) for x, y in combinations(lab_cells, 2)] + [(lab_cells[0], lab_cells[0])]
    outcomes = [o for res in await asyncio.gather(*[pair_cell(x, y, lab_suite.task_ids) for x, y in pairs]) for o in res]
    real = [o for o in outcomes if not o[1].endswith("'")]
    elo = pref.elo_with_ci(real) if real else {}
    win = {f"{x.checkpoint} vs {y.checkpoint}": pref.win_rate([o[2] for o in outcomes
                                                               if o[0] == x.checkpoint and o[1].rstrip("'") == y.checkpoint])
           for x, y in pairs}
    table = [{"checkpoint": l.checkpoint, "step": l.step, **{f"lab_{k}": v for k, v in l.metrics.items() if k != "finish"},
              **{f"cuad_{k}": v for k, v in c.metrics.items()}, "elo": elo.get(l.checkpoint)}
             for l, c in zip(lab_cells, cuad_cells)]
    await flyte.report.replace.aio(f"<h3>checkpoint matrix ({(time.time() - t0) / 60:.0f} min)</h3>"
                                   f"<pre>{json.dumps(table, indent=1)}</pre><pre>{json.dumps(win, indent=1)}</pre>",
                                   do_flush=True)
    slim = [c.model_copy(update={"rollouts": []}) for c in lab_cells + cuad_cells]
    return Matrix(cells=slim, elo=elo, win_rates=win, table=table)


if __name__ == "__main__":
    import argparse

    import flyte.remote

    from sweep import pick_tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--eval-tasks", type=int, default=48)
    args = ap.parse_args()
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    tr = flyte.remote.Run.get(name=args.train_run).outputs()[0]
    cks = [Checkpoint(step=0)] + [Checkpoint(step=s, adapter=d) for s, d in sorted(tr.checkpoints.items())]
    held = pick_tasks(json.load(open("lab_tasks.json"))["tasks"], args.eval_tasks)
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(
        matrix, cks, Suite(kind="lab", task_ids=held), Suite(kind="cuad"))
    print(run.name, run.url)
