# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""The two probes that need no training: padding, and a pairwise tournament, both over rollouts you already have.

``make_variants``  pads real deliverables with plausible but non-responsive issues (``padding.py``'s prompts) and
                   writes them out as rollouts, so ``grade_shards.py`` grades them like any other rollout.
``pairs``          judges deliverable pairs in both orders with ``preference.PREFERENCE_PROMPT``: padded against
                   unpadded, policy against policy, and a policy against itself (the A/A test).

Both run on one GPU each, and return compact results inline.

    python probes.py variants --rollouts sweep_rollouts.jsonl.gz
    python probes.py pairs --rollouts sweep_rollouts.jsonl.gz --variants variants.jsonl.gz
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import tempfile
from pathlib import Path

import flyte
import flyte.io
from pydantic import BaseModel

import lab_harness as lh
import preference as pref
import servers
from config import PROPAGATE, gpu_env_kwargs
from images import LAB_COMMIT
from padding import GENERIC_PROMPT, SPECULATIVE_PROMPT, _pandoc

probe_env = flyte.TaskEnvironment(name="legal-rl-probe", env_vars=PROPAGATE, **gpu_env_kwargs())


def _deliverable_md(outputs_b64: str, name: str | None) -> str | None:
    with tempfile.TemporaryDirectory() as d:
        out = lh.untar_b64(outputs_b64, Path(d))
        docx = next((p for p in out.rglob("*.docx") if p.name == name), None) or next(out.rglob("*.docx"), None)
        return _pandoc([str(docx), "-t", "gfm"]) if docx else None


def _md_to_outputs(md: str, name: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        _pandoc(["-f", "gfm", "-o", str(Path(d) / name)], stdin=md)
        return lh._tar_b64(Path(d))


class VariantSet(BaseModel):
    data: str  # base64 gzip of rollouts.jsonl: every variant as a Rollout, rollout_id "<source>::<variant>"
    n: int
    added_chars: dict[str, list[int]]


@probe_env.task(cache="auto", retries=3)
async def make_variants(rollouts: flyte.io.File, policy: str = "gpt-oss-120b", per_task: int = 2,
                        writer_hf: str = "openai/gpt-oss-20b") -> VariantSet:
    import base64

    import openai

    with gzip.open(await rollouts.download(), "rt") as f:
        rs = [lh.Rollout.model_validate_json(l) for l in f]
    rs = [r for r in rs if r.policy == policy and r.deliverable_chars > 0 and r.finish_reason not in lh.TRUNCATED]
    picked = [r for t in sorted({r.task_id for r in rs})
              for r in sorted((x for x in rs if x.task_id == t), key=lambda x: x.sample)[:per_task]]
    lab = lh.fetch_lab(sorted({r.task_id for r in picked}), LAB_COMMIT, Path("/tmp/lab"))
    srv = servers.ServerSpec(served_name="writer", hf_id=writer_hf, gpus=[0], max_model_len=65536,
                             gpu_memory_utilization=0.9)
    await servers.ensure(srv)
    client = openai.AsyncOpenAI(base_url=srv.url, api_key="x", timeout=900)
    sem = asyncio.Semaphore(32)
    out: list[lh.Rollout] = []
    added: dict[str, list[int]] = {"generic": [], "speculative": []}

    async def write(prompt: str) -> str:
        async with sem:
            r = await client.chat.completions.create(model="writer", temperature=0.7, max_tokens=6000,
                                                     messages=[{"role": "user", "content": prompt}],
                                                     extra_body={"reasoning_effort": "low"})
            return r.choices[0].message.content or ""

    async def one(r: lh.Rollout):
        task, _ = lh.load_task(lab, r.task_id)
        name = next(iter(task.deliverables))
        md = await asyncio.to_thread(_deliverable_md, r.outputs_b64, name)
        if not md:
            return
        gen, spec = await asyncio.gather(write(GENERIC_PROMPT.format(instructions=task.instructions)),
                                         write(SPECULATIVE_PROMPT.format(instructions=task.instructions,
                                                                         draft=md[:60_000])))
        for variant, extra in (("roundtrip", ""), ("generic", gen), ("speculative", spec)):
            body = md if not extra else md.rstrip() + "\n\n" + extra.strip() + "\n"
            if extra:
                added[variant].append(len(body) - len(md))
            b64 = await asyncio.to_thread(_md_to_outputs, body, name)
            out.append(r.model_copy(update={"rollout_id": f"{r.rollout_id}::{variant}", "outputs_b64": b64,
                                            "deliverable_chars": len(body), "tokens": []}))

    await asyncio.gather(*[one(r) for r in picked])
    blob = base64.b64encode(gzip.compress("".join(r.model_dump_json() + "\n" for r in out).encode())).decode()
    return VariantSet(data=blob, n=len(out), added_chars=added)


class PairSpec(BaseModel):
    kind: str  # padding | policy | aa
    task_id: str
    a: str  # rollout ids
    b: str


@probe_env.task(cache="auto", retries=3)
async def pairs(rollouts: flyte.io.File, specs: list[PairSpec], judge: lh.JudgeSpec) -> list[dict]:
    """Every pair in both orders. Returns {kind, task_id, a, b, score (a's share), first_wins}."""
    import openai

    rows: dict[str, lh.Rollout] = {}
    with gzip.open(await rollouts.download(), "rt") as f:
        for l in f:
            r = lh.Rollout.model_validate_json(l)
            rows[r.rollout_id] = r
    lab = lh.fetch_lab(sorted({s.task_id for s in specs}), LAB_COMMIT, Path("/tmp/lab"))
    srv = servers.ServerSpec(served_name=judge.name, hf_id=judge.hf_id, gpus=[0], max_model_len=65536,
                             gpu_memory_utilization=0.9)
    await servers.ensure(srv)
    client = openai.OpenAI(base_url=srv.url, api_key="x", timeout=900)
    sem = asyncio.Semaphore(24)
    md_cache: dict[str, str] = {}

    def md(rid: str, name: str) -> str:
        if rid not in md_cache:
            md_cache[rid] = _deliverable_md(rows[rid].outputs_b64, name) or "(no deliverable)"
        return md_cache[rid]

    async def one(s: PairSpec):
        task, _ = lh.load_task(lab, s.task_id)
        name = next(iter(task.deliverables))
        async with sem:
            try:
                res = await asyncio.to_thread(pref.both_orders, client, judge.name, task.instructions,
                                              md(s.a, name), md(s.b, name), reasoning_effort=judge.reasoning_effort,
                                              thinking=judge.thinking)
            except Exception as e:  # noqa: BLE001
                return {**s.model_dump(), "error": str(e)[:200]}
        return {**s.model_dump(), "score": res["score"], "first_wins": res["first_wins"]}

    return list(await asyncio.gather(*[one(s) for s in specs]))


def pair_specs(rollouts: list[dict], variants: list[dict] | None, seed: int = 0) -> list[PairSpec]:
    """Padding pairs (padded vs roundtrip), policy pairs (same task, same sample index, different policies), and
    A/A pairs (same policy and task, samples 0 and 1)."""
    rng = random.Random(seed)
    specs: list[PairSpec] = []
    for v in variants or []:
        src, variant = v["rollout_id"].rsplit("::", 1)
        if variant != "roundtrip":
            specs.append(PairSpec(kind=f"padding-{variant}", task_id=v["task_id"], a=v["rollout_id"],
                                  b=f"{src}::roundtrip"))
    by = {(r["policy"], r["task_id"], r["sample"]): r for r in rollouts if r["deliverable_chars"] > 0}
    pols = sorted({r["policy"] for r in rollouts})
    tasks = sorted({r["task_id"] for r in rollouts})
    for t in tasks:
        for k in (0, 1):
            for i, p in enumerate(pols):
                for q in pols[i + 1:]:
                    if (p, t, k) in by and (q, t, k) in by:
                        a, b = by[(p, t, k)]["rollout_id"], by[(q, t, k)]["rollout_id"]
                        a, b = (a, b) if rng.random() < 0.5 else (b, a)
                        specs.append(PairSpec(kind="policy", task_id=t, a=a, b=b))
        for p in pols:
            if (p, t, 0) in by and (p, t, 1) in by:
                specs.append(PairSpec(kind="aa", task_id=t, a=by[(p, t, 0)]["rollout_id"], b=by[(p, t, 1)]["rollout_id"]))
    return specs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["variants", "pairs"])
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--variants", default=None, help="variants.jsonl.gz from the variants step")
    ap.add_argument("--judge", default="gpt-oss-20b")
    args = ap.parse_args()
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    if args.what == "variants":
        run = flyte.run(make_variants, flyte.io.File.from_local_sync(args.rollouts))
    else:
        base = [json.loads(l) for l in gzip.open(args.rollouts, "rt")]
        var = [json.loads(l) for l in gzip.open(args.variants, "rt")] if args.variants else None
        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
            with gzip.open(tmp.name, "wt") as g:
                for r in base + (var or []):
                    g.write(json.dumps(r) + "\n")
        judges = {"gpt-oss-20b": lh.JudgeSpec(name="gpt-oss-20b", hf_id="openai/gpt-oss-20b", reasoning_effort="low"),
                  "nemotron-3-nano-30b-fp8": lh.JudgeSpec(name="nemotron-3-nano-30b-fp8",
                                                          hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                                                          thinking=False)}
        run = flyte.run(pairs, flyte.io.File.from_local_sync(tmp.name), pair_specs(base, var), judges[args.judge])
    print(run.name, run.url)
