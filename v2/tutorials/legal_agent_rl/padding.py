# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte>=2.13.1", "pydantic", "kubernetes"]
# ///
"""Does the rubric pay for padding? Add plausible but non-responsive issues to real deliverables, then ask the rubric
and a pairwise judge what changed.

Harvey's post notes that rubric gains don't reliably buy preference, and that outputs grew in training with no
length penalty. If a pass/fail rubric pays for coverage, padding a memo with extra issues should raise its rubric
score (more chances to hit a criterion) while a reader who values judgment prefers the original. This tests that
directly, with no training:

    roundtrip     the deliverable converted to Markdown and back, unchanged (the control: same formatting as below)
    generic       + four general cautions any memo of this kind could carry, written without seeing the matter
    speculative   + four issues that name this matter's parties but rest on no fact in the draft or the documents

Each variant is graded with LAB's rubric by two open judges, and compared with ``roundtrip`` in both orders.

    CLUSTER=amd GPU_QUEUE=<queue> python padding.py --sweep-run <run name>
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.report
from pydantic import BaseModel, Field

import lab_harness as lh
import preference as pref
import servers
from config import PROPAGATE, gpu_env_kwargs
from images import LAB_COMMIT

pad_env = flyte.TaskEnvironment(name="legal-rl-padding", env_vars=PROPAGATE, **gpu_env_kwargs())

GENERIC_PROMPT = """A colleague is finishing a legal work product for this request:

{instructions}

Write four short additional considerations (one paragraph each, 60-90 words) that a cautious lawyer might append to
almost any work product of this type. Keep them general: do not refer to specific parties, documents, dates or
amounts. Output Markdown: a level-2 heading "Additional Considerations" and four numbered paragraphs with bold lead-ins."""

SPECULATIVE_PROMPT = """Here is a draft legal work product and the request it answers.

## Request
{instructions}

## Draft
{draft}

Write four additional issues (one paragraph each, 60-90 words) that sound specific to this matter: use the parties'
names and the kinds of documents in the draft. Each issue must be speculative: it must NOT rest on any fact stated in
the draft, and it must hedge ("may", "could", "should be confirmed"). Do not repeat issues the draft already raises.
Output Markdown: a level-2 heading "Further Issues for Review" and four numbered paragraphs with bold lead-ins."""


class PaddingConfig(BaseModel):
    policy: str = "gpt-oss-120b"  # whose deliverables to pad
    per_task: int = 2
    judges: list[lh.JudgeSpec] = [
        lh.JudgeSpec(name="gpt-oss-120b", hf_id="openai/gpt-oss-120b", reasoning_effort="low"),
        lh.JudgeSpec(name="nemotron-3-nano-30b", hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
    ]
    writer: str = "gpt-oss-120b"


class PaddingRow(BaseModel):
    rollout_id: str
    task_id: str
    variant: str
    chars: int
    terms: dict[str, dict[str, float]] = Field(default_factory=dict)  # judge -> c/h/a
    pref_vs_roundtrip: dict[str, dict] = Field(default_factory=dict)  # judge -> both_orders result


class PaddingResult(BaseModel):
    rows: list[PaddingRow]
    summary: dict


def _pandoc(args: list[str], stdin: str | None = None) -> str:
    return subprocess.run(["pandoc", *args], input=stdin, capture_output=True, text=True, check=True).stdout


def _variant_outputs(rollout: dict, task: lh.LabTask, extra_md: str | None) -> tuple[str, str]:
    """(outputs_b64 for grading, markdown text for the preference judge) with ``extra_md`` appended."""
    with tempfile.TemporaryDirectory() as d:
        out = lh.untar_b64(rollout["outputs_b64"], Path(d) / "output")
        name = next(iter(task.deliverables))
        src = next((p for p in out.rglob("*.docx") if p.name == name), None) or next(out.rglob("*.docx"), None)
        if src is None:
            raise FileNotFoundError("no .docx deliverable")
        md = _pandoc([str(src), "-t", "gfm"])
        if extra_md:
            md = md.rstrip() + "\n\n" + extra_md.strip() + "\n"
        src.unlink()
        _pandoc(["-f", "gfm", "-o", str(out / name)], stdin=md)
        return lh._tar_b64(out), md


@flyte.trace
async def traced_text(base_url: str, model: str, prompt: str) -> str:
    import openai

    client = openai.AsyncOpenAI(base_url=base_url, api_key="x", timeout=900)
    r = await client.chat.completions.create(model=model, temperature=0.7, max_tokens=6000,
                                             messages=[{"role": "user", "content": prompt}],
                                             extra_body={"reasoning_effort": "low"})
    return r.choices[0].message.content or ""


@flyte.trace
async def traced_pref(base_url: str, model: str, instructions: str, a: str, b: str, effort: str | None) -> dict:
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="x", timeout=900)
    return await asyncio.to_thread(pref.both_orders, client, model, instructions, a, b, reasoning_effort=effort)


@pad_env.task(cache="auto", retries=3, report=True)
async def padding_probe(sweep_data: flyte.io.File, cfg: PaddingConfig) -> PaddingResult:
    t0 = time.time()
    local = await sweep_data.download()
    rollouts, groups = [], {}
    with gzip.open(local, "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] == "groups":
                groups[row["task_id"]] = row["groups"]
            elif row["policy"] == cfg.policy and row["deliverable_chars"] > 0 and row["finish_reason"] not in lh.TRUNCATED:
                rollouts.append(row)
    picked: list[dict] = []
    for t in sorted({r["task_id"] for r in rollouts}):
        picked += sorted((r for r in rollouts if r["task_id"] == t), key=lambda r: r["sample"])[:cfg.per_task]
    lab = lh.fetch_lab(sorted({r["task_id"] for r in picked}), LAB_COMMIT, Path("/tmp/lab"))

    S = servers.ServerSpec
    srvs = {"gpt-oss-120b": S(served_name="gpt-oss-120b", hf_id="openai/gpt-oss-120b", gpus=[0, 1, 2, 3],
                              max_model_len=131072),
            "nemotron-3-nano-30b": S(served_name="nemotron-3-nano-30b", hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                                     gpus=[4, 5, 6, 7], max_model_len=131072)}
    await asyncio.gather(*[servers.ensure(s) for s in srvs.values()])
    writer = srvs[cfg.writer]
    sem = asyncio.Semaphore(32)

    async def one(r: dict) -> list[PaddingRow]:
        async with sem:
            task, _ = lh.load_task(lab, r["task_id"])
            base_b64, base_md = _variant_outputs(r, task, None)
            generic = await traced_text(writer.url, writer.served_name,
                                        GENERIC_PROMPT.format(instructions=task.instructions))
            spec = await traced_text(writer.url, writer.served_name,
                                     SPECULATIVE_PROMPT.format(instructions=task.instructions, draft=base_md[:60_000]))
            variants = {"roundtrip": (base_b64, base_md)}
            for name, extra in (("generic", generic), ("speculative", spec)):
                variants[name] = _variant_outputs(r, task, extra)
            rows = []
            for name, (b64, md) in variants.items():
                row = PaddingRow(rollout_id=r["rollout_id"], task_id=r["task_id"], variant=name, chars=len(md))
                fake = lh.Rollout(rollout_id=f"{r['rollout_id']}::{name}", task_id=r["task_id"], policy=cfg.policy,
                                  harness=r["harness"], sample=r["sample"], finish_reason=r["finish_reason"],
                                  turns=r["turns"], input_tokens=0, output_tokens=0, wall_s=0, outputs_b64=b64)
                for j in cfg.judges:
                    jj = lh.OpenAICompatJudge(j, srvs[j.name].url)
                    v = await asyncio.to_thread(lh.grade, fake, task, jj, j.name, 16)
                    row.terms[j.name] = lh.reward_terms(v.verdicts, groups.get(r["task_id"], {}))
                    if name != "roundtrip":
                        row.pref_vs_roundtrip[j.name] = await traced_pref(srvs[j.name].url, j.name, task.instructions,
                                                                          md, base_md, j.reasoning_effort)
                rows.append(row)
            return rows

    rows = [x for rs in await asyncio.gather(*[one(r) for r in picked], return_exceptions=True)
            if not isinstance(rs, BaseException) for x in rs]
    summary = summarize(rows, [j.name for j in cfg.judges])
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    await flyte.report.replace.aio(f"<pre>{json.dumps(summary, indent=1)}</pre>", do_flush=True)
    return PaddingResult(rows=rows, summary=summary)


# {{docs-fragment summarize}}
def summarize(rows: list[PaddingRow], judges: list[str]) -> dict:
    """Per variant and judge: the change in each rubric term against the roundtrip control (paired by rollout), and
    the padded draft's win rate against it (0.5 = no preference)."""
    import statistics as st

    base = {r.rollout_id: r for r in rows if r.variant == "roundtrip"}
    out: dict = {"pairs": len(base)}
    for variant in ("generic", "speculative"):
        vs = [r for r in rows if r.variant == variant and r.rollout_id in base]
        out[variant] = {"n": len(vs), "chars_added_median": st.median(r.chars - base[r.rollout_id].chars for r in vs)
                        if vs else None}
        for j in judges:
            d = {k: [r.terms[j][k] - base[r.rollout_id].terms[j][k] for r in vs if j in r.terms
                     and j in base[r.rollout_id].terms] for k in ("c", "h", "a")}
            scores = [r.pref_vs_roundtrip[j]["score"] for r in vs if j in r.pref_vs_roundtrip]
            out[variant][j] = {
                "delta": {k: {"mean": st.fmean(x), "up": sum(v > 0 for v in x), "down": sum(v < 0 for v in x)}
                          for k, x in d.items() if x},
                "padded_win_rate": pref.win_rate(scores),
                "position_first_wins": st.fmean(r.pref_vs_roundtrip[j]["first_wins"] / 2 for r in vs
                                                if j in r.pref_vs_roundtrip) if scores else None}
    return out
# {{/docs-fragment summarize}}


if __name__ == "__main__":
    import argparse

    import flyte.remote

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-run", required=True)
    args = ap.parse_args()
    flyte.init_from_config(os.getenv("FLYTE_CONFIG"))
    sweep_out = flyte.remote.Run.get(name=args.sweep_run).outputs()[0]
    run = flyte.with_runcontext(queue=os.getenv("GPU_QUEUE") or None).run(padding_probe, sweep_out.data, PaddingConfig())
    print(run.name, run.url)
