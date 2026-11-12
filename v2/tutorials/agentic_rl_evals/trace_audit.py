# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "flyte>=2.13.0",
#    "huggingface_hub>=0.34",
#    "anthropic>=0.70",
#    "pydantic>=2",
# ]
# main = "audit"
# params = ""
# ///
"""Audit published agentic-RL eval traces: what the trajectories say before you read the loss curve.

Mercor published the eval traces behind its 397B SkyRL training guide (``mercor/ApexAgentsRecipe-*``
on Hugging Face). This pipeline reads a trace set end to end:

1. ``fetch_set``      one task per trace set; downloads it once (``cache="auto"``) into a ``flyte.io.Dir``.
2. ``featurize``      deterministic features for every trajectory: outcome, exception, wall-clock,
                      tokens, turns, tests passed vs total, tool-output error signatures. No LLM.
3. ``classify_chunk`` an LLM (Claude) labels each *failed* trajectory with one category from a fixed
                      taxonomy. Chunks of ~20 trajectories per task; each call is ``@flyte.trace``d so a
                      retried chunk replays the calls it already paid for. The classifier (model +
                      prompt version + taxonomy) is an input, so changing it re-labels; nothing else re-runs.
4. ``audit``          the driver: fans out per set, then per chunk, and writes a report.

Run::

    flyte run trace_audit.py audit                       # all four Terminal-Bench 2.1 sets
    flyte run trace_audit.py audit --classify false      # features only, no LLM spend

Needs a Flyte/Union backend and an Anthropic API key secret named ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import flyte
import flyte.io
import flyte.report
from pydantic import BaseModel

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(name="agentic-rl-trace-audit", python_version=(3, 12)).with_pip_packages(
    "huggingface_hub>=0.34", "anthropic>=0.70", "pydantic>=2"
)
# {{/docs-fragment image}}

# The secret holding an Anthropic API key; override with ANTHROPIC_SECRET=<name> when you launch.
ANTHROPIC_SECRET = os.environ.get("ANTHROPIC_SECRET", "ANTHROPIC_API_KEY")
TBENCH_REPO = "mercor/ApexAgentsRecipe-TBench2_1-EvalTraces"
TBENCH_SETS = [
    "Qwen3p6-35B_untrained_traces",
    "Qwen3p6-35B_trained_traces",
    "Qwen3p5-397B_untrained_traces",
    "Qwen3p5-397B_trained_traces",
]

# {{docs-fragment envs}}
fetch_env = flyte.TaskEnvironment(
    name="trace-audit-fetch",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="20Gi"),
)
classify_env = flyte.TaskEnvironment(
    name="trace-audit-classifier",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=[flyte.Secret(key=ANTHROPIC_SECRET, as_env_var="ANTHROPIC_API_KEY")],
    env_vars={"ANTHROPIC_SECRET": ANTHROPIC_SECRET},  # resolve the same secret name in the container
)
driver_env = flyte.TaskEnvironment(
    name="trace-audit-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    # child task specs are built inside this container, so it needs the same secret name to hand down
    env_vars={"ANTHROPIC_SECRET": ANTHROPIC_SECRET},
    depends_on=[fetch_env, classify_env],
)
# {{/docs-fragment envs}}


# --- types -------------------------------------------------------------------------------------------


# {{docs-fragment types}}
class TraceFeatures(BaseModel):
    """Everything we can say about one trajectory without an LLM."""

    trace_set: str
    trajectory_id: str
    task_name: str
    score: float
    outcome: str  # completed / error / timeout / ... as recorded by the harness
    exception_type: str | None
    duration_s: float
    assistant_turns: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    max_prompt_tokens: int
    tests_passed: int
    tests_total: int
    empty_response_loop: bool
    tool_error_signatures: dict[str, int]  # signature -> count of tool outputs that matched it


class Classifier(BaseModel):
    """The LLM labeler is an *input*: change any field and only the labels re-run."""

    model: str = "claude-sonnet-5-5"
    prompt_version: str = "v1"
    taxonomy: list[str] = [
        "env_setup",  # missing tool/package/runtime the task assumed, install fights
        "tool_or_harness",  # the harness/terminal/tool misbehaved (truncation, hangs, garbled output)
        "budget_exhausted",  # ran out of steps/time/context while still making progress
        "wrong_approach",  # solved a different problem or picked a method that can't work
        "near_miss",  # mostly right; failed a small number of tests on details
        "false_completion",  # declared done without verifying; tests show it isn't
        "gave_up",  # stopped early, refused, or looped without progress
        "grader_suspect",  # the work looks right and the tests/grader look wrong or brittle
    ]


class Label(BaseModel):
    trajectory_id: str
    category: str
    evidence: str  # one sentence quoting or paraphrasing the trace


class Digest(BaseModel):
    """A compact, LLM-sized view of one failed trajectory."""

    trajectory_id: str
    trace_set: str
    text: str


class SetSummary(BaseModel):
    trace_set: str
    n: int
    pass_at_1: float
    outcomes: dict[str, int]
    duration_p50_s: float
    duration_p90_s: float
    duration_max_s: float
    longest_decile_share_of_time: float  # fraction of all wall-clock spent in the slowest 10%
    straggler_ratio_p50: float  # median over tasks of (slowest rollout / fastest rollout)
    prompt_tokens_mean: float
    cached_fraction: float
    completion_tokens_mean: float
    failed_with_most_tests_passing: int  # score 0 but >= 75% of tests passed
    failed_total: int
    tool_error_signatures: dict[str, int]
    labels: dict[str, int]


# {{/docs-fragment types}}
# --- 1. fetch ------------------------------------------------------------------------------------------


# {{docs-fragment fetch}}
@fetch_env.task(cache="auto", retries=3)
async def fetch_set(repo_id: str, trace_set: str) -> flyte.io.Dir:
    """Download one trace set once; every later run (and every fork) reuses the cached Dir."""
    import tempfile

    from huggingface_hub import snapshot_download

    local = tempfile.mkdtemp(prefix="traces-")
    await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{trace_set}/**"],
        local_dir=local,
        max_workers=16,
    )
    return await flyte.io.Dir.from_local(str(Path(local) / trace_set))


# {{/docs-fragment fetch}}
# --- 2. featurize ------------------------------------------------------------------------------------


# Signatures of tool output that usually mean the environment or harness, not the policy, went wrong.
# {{docs-fragment signatures}}
TOOL_ERROR_SIGNATURES = {
    "command_not_found": re.compile(r"command not found|No such file or directory: '?/usr/bin"),
    "python_traceback": re.compile(r"Traceback \(most recent call last\)"),
    "module_not_found": re.compile(r"ModuleNotFoundError|No module named"),
    "permission_denied": re.compile(r"Permission denied"),
    "timed_out": re.compile(r"timed out|Timeout|TimeoutError", re.IGNORECASE),
    "killed_oom": re.compile(r"\bKilled\b|Out of memory|MemoryError"),
    "apt_or_pip_failure": re.compile(r"E: Unable to locate package|Could not find a version that satisfies"),
    "network": re.compile(r"Could not resolve host|Temporary failure in name resolution|Connection refused"),
}
# {{/docs-fragment signatures}}


def _features(trace_set: str, t: dict) -> TraceFeatures:
    out = t.get("trajectory_output") or {}
    usage = out.get("usage_metrics") or {}
    msgs = t.get("trajectory_messages") or []
    sig_counts: Counter = Counter()
    for m in msgs:
        if m.get("role") != "tool":
            continue
        content = str(m.get("content") or "")
        for name, rx in TOOL_ERROR_SIGNATURES.items():
            if rx.search(content):
                sig_counts[name] += 1
    return TraceFeatures(
        trace_set=trace_set,
        trajectory_id=t["trajectory_id"],
        task_name=t.get("task_name", ""),
        score=float(out.get("score") or 0.0),
        outcome=str(out.get("outcome") or t.get("trajectory_status") or "unknown"),
        exception_type=out.get("exception_type"),
        duration_s=float(out.get("duration_seconds") or 0.0),
        assistant_turns=sum(1 for m in msgs if m.get("role") == "assistant"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=int(usage.get("cached_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        max_prompt_tokens=int(usage.get("max_prompt_tokens") or 0),
        tests_passed=int(out.get("tests_passed") or 0),
        tests_total=int(out.get("tests_total") or 0),
        empty_response_loop=bool(out.get("empty_response_loop")),
        tool_error_signatures=dict(sig_counts),
    )


def _digest(trace_set: str, t: dict, max_chars: int = 9000) -> Digest:
    """Task prompt head, the last few turns, the failing tests and the final answer."""
    out = t.get("trajectory_output") or {}
    msgs = t.get("trajectory_messages") or []
    task_prompt = str(msgs[0].get("content", ""))[:1500] if msgs else ""
    tail = []
    for m in msgs[-8:]:
        tail.append(f"[{m.get('role')}] {str(m.get('content') or '')[:700]}")
    failing = [k for k, v in (out.get("test_statuses") or {}).items() if v != "pass"]
    text = "\n".join(
        [
            f"TASK: {t.get('task_name')}",
            f"OUTCOME: {out.get('outcome')} | exception: {out.get('exception_type')} | "
            f"duration_s: {out.get('duration_seconds')} | tests {out.get('tests_passed')}/{out.get('tests_total')}",
            f"FAILING TESTS: {failing[:12]}",
            f"TEST OUTPUT TAIL: {str((out.get('test_summary_metadata') or {}).get('test_stdout_tail', ''))[-800:]}",
            "TASK PROMPT (head):",
            task_prompt,
            "LAST TURNS:",
            *tail,
            f"FINAL ANSWER: {str(t.get('final_answer') or '')[:800]}",
        ]
    )
    return Digest(trajectory_id=t["trajectory_id"], trace_set=trace_set, text=text[:max_chars])


# {{docs-fragment featurize}}
@fetch_env.task(cache="auto")
async def featurize(trace_set: str, traces: flyte.io.Dir) -> tuple[list[TraceFeatures], list[Digest]]:
    """Deterministic pass over every trajectory in the set; digests only for the failures."""
    local = Path(await traces.download())
    feats: list[TraceFeatures] = []
    digests: list[Digest] = []
    for path in sorted(local.rglob("traj_*.json")):
        t = json.loads(path.read_text())
        t = t.get("trajectory", t)
        f = _features(trace_set, t)
        feats.append(f)
        if f.score < 1.0:
            digests.append(_digest(trace_set, t))
    return feats, digests


# {{/docs-fragment featurize}}
# --- 3. classify -------------------------------------------------------------------------------------


SYSTEM_PROMPT = """You label failed agent trajectories from a terminal-task benchmark (Terminal-Bench).
Pick exactly one category from the taxonomy for the *primary* reason the trajectory failed, and give one
sentence of evidence that quotes or closely paraphrases the trace. Prefer environment/harness categories only
when the trace shows the environment or tools misbehaving, not when the agent merely made a mistake."""


# {{docs-fragment classify}}
@flyte.trace
async def call_classifier(digest: Digest, classifier: Classifier) -> Label:
    """One LLM call. Traced: if the chunk task retries, finished calls replay instead of re-billing."""
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=4)
    schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": classifier.taxonomy},
            "evidence": {"type": "string"},
        },
        "required": ["category", "evidence"],
        "additionalProperties": False,
    }
    resp = await client.messages.create(
        model=classifier.model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": f"Taxonomy: {classifier.taxonomy}\n\n{digest.text}"}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text) if resp.stop_reason != "refusal" else {"category": "unlabeled", "evidence": "refusal"}
    return Label(trajectory_id=digest.trajectory_id, category=data["category"], evidence=data["evidence"])


@classify_env.task(
    cache="auto",
    # 429s past the SDK's own retries come back here: back off, then retry the chunk; finished calls replay
    retries=flyte.RetryStrategy(
        count=3, backoff=flyte.Backoff(base=timedelta(seconds=20), factor=2.0, cap=timedelta(minutes=3))
    ),
)
async def classify_chunk(digests: list[Digest], classifier: Classifier, max_concurrency: int = 8) -> list[Label]:
    """~20 trajectories per task: big enough that pod startup is noise, small enough to retry cheaply."""
    sem = asyncio.Semaphore(max_concurrency)

    async def one(d: Digest) -> Label:
        async with sem:
            return await call_classifier(d, classifier)

    return list(await asyncio.gather(*(one(d) for d in digests)))


# {{/docs-fragment classify}}
# --- 4. aggregate + driver -----------------------------------------------------------------------------


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(trace_set: str, feats: list[TraceFeatures], labels: list[Label]) -> SetSummary:
    durs = [f.duration_s for f in feats]
    top = sorted(durs, reverse=True)[: max(1, len(durs) // 10)]
    by_task: dict[str, list[float]] = defaultdict(list)
    for f in feats:
        by_task[f.task_name].append(f.duration_s)
    ratios = [max(v) / max(min(v), 1.0) for v in by_task.values() if len(v) > 1]
    failed = [f for f in feats if f.score < 1.0]
    prompt = sum(f.prompt_tokens for f in feats)
    sigs: Counter = Counter()
    for f in feats:
        sigs.update({k: 1 for k in f.tool_error_signatures})  # trajectories with >= 1 match
    return SetSummary(
        trace_set=trace_set,
        n=len(feats),
        pass_at_1=round(statistics.fmean(f.score for f in feats), 4) if feats else 0.0,
        outcomes=dict(Counter(f.outcome for f in feats)),
        duration_p50_s=round(_pct(durs, 0.5), 1),
        duration_p90_s=round(_pct(durs, 0.9), 1),
        duration_max_s=round(max(durs, default=0.0), 1),
        longest_decile_share_of_time=round(sum(top) / max(sum(durs), 1.0), 3),
        straggler_ratio_p50=round(statistics.median(ratios), 2) if ratios else 0.0,
        prompt_tokens_mean=round(prompt / max(len(feats), 1)),
        cached_fraction=round(sum(f.cached_tokens for f in feats) / max(prompt, 1), 3),
        completion_tokens_mean=round(sum(f.completion_tokens for f in feats) / max(len(feats), 1)),
        failed_with_most_tests_passing=sum(
            1 for f in failed if f.tests_total and f.tests_passed / f.tests_total >= 0.75
        ),
        failed_total=len(failed),
        tool_error_signatures=dict(sigs.most_common()),
        labels=dict(Counter(lb.category for lb in labels).most_common()),
    )


def render(summaries: list[SetSummary], examples: dict[str, list[Label]]) -> str:
    rows = []
    for s in summaries:
        rows.append(
            f"<tr><td>{s.trace_set}</td><td>{s.n}</td><td>{s.pass_at_1:.1%}</td>"
            f"<td>{s.duration_p50_s / 60:.0f} / {s.duration_p90_s / 60:.0f} / {s.duration_max_s / 60:.0f}</td>"
            f"<td>{s.longest_decile_share_of_time:.0%}</td><td>{s.straggler_ratio_p50}×</td>"
            f"<td>{s.prompt_tokens_mean / 1e6:.2f}M ({s.cached_fraction:.0%} cached)</td>"
            f"<td>{s.failed_with_most_tests_passing}/{s.failed_total}</td>"
            f"<td>{s.outcomes}</td><td>{s.labels}</td><td>{s.tool_error_signatures}</td></tr>"
        )
    ex = "".join(
        f"<h4>{cat}</h4><ul>" + "".join(f"<li><code>{lb.trajectory_id}</code>: {lb.evidence}</li>" for lb in lbs[:4]) + "</ul>"
        for cat, lbs in examples.items()
    )
    return (
        "<h2>Trace audit</h2><table border=1 cellpadding=4><tr><th>set</th><th>n</th><th>pass@1</th>"
        "<th>minutes p50/p90/max</th><th>time in slowest 10%</th><th>straggler ratio (median task)</th>"
        "<th>prompt tokens / traj</th><th>failed but ≥75% tests pass</th><th>outcomes</th><th>failure labels</th>"
        f"<th>tool-output error signatures (trajectories)</th></tr>{''.join(rows)}</table>"
        f"<h3>Example labels</h3>{ex}"
    )


# {{docs-fragment driver}}
@driver_env.task(report=True)
async def audit(
    repo_id: str = TBENCH_REPO,
    trace_sets: list[str] = TBENCH_SETS,
    classifier: Classifier = Classifier(),
    classify: bool = True,
    chunk_size: int = 20,
) -> list[SetSummary]:
    async def one_set(trace_set: str) -> tuple[SetSummary, list[Label]]:
        with flyte.group(trace_set):
            traces = await fetch_set(repo_id, trace_set)
            feats, digests = await featurize(trace_set, traces)
            labels: list[Label] = []
            if classify and digests:
                chunks = [digests[i : i + chunk_size] for i in range(0, len(digests), chunk_size)]
                results = await asyncio.gather(
                    *(classify_chunk(c, classifier) for c in chunks), return_exceptions=True
                )
                failed = [r for r in results if not isinstance(r, list)]
                for r in results:  # a chunk that still fails after retries is a gap, not a crash...
                    if isinstance(r, list):
                        labels.extend(r)
                if len(failed) == len(chunks):  # ...unless every chunk failed: that's a broken classifier
                    raise RuntimeError(f"all {len(chunks)} classify chunks failed, e.g. {failed[0]!r}")
            return summarize(trace_set, feats, labels), labels

    pairs = await asyncio.gather(*(one_set(s) for s in trace_sets))
    summaries = [p[0] for p in pairs]
    examples: dict[str, list[Label]] = defaultdict(list)
    for _, labels in pairs:
        for lb in labels:
            examples[lb.category].append(lb)
    await flyte.report.replace.aio(render(summaries, examples), do_flush=True)
    return summaries


# {{/docs-fragment driver}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(audit)
    print(run.url)
