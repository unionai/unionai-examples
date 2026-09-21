"""The experiment: same battery, same cases, same downstream logic, two arms.

    with System 1     — TypeSafe answers the battery, System 2 writes the review
    without System 1  — System 2 answers the battery *and* writes the review

Both arms owe the same artifact — eighteen typed answers plus a review — and both
are composed and routed by the identical code in `battery.py`. The only thing
that changes is where the answers come from. Every unit is an independent Flyte
action, so the matrix fans out across the cluster.

    flyte run benchmark.py run_benchmark
    flyte run benchmark.py run_benchmark --repeats 3 --num_cases 8
"""

from __future__ import annotations

import asyncio
import pathlib
import statistics

import flyte
import flyte.report
from _runtime import driver_env, env
from battery import CASES, ReviewBattery, case, compose, fired, route
from flyteplugins.typesafe_ai import ask_with_info
from system2 import PRICING, System2

ARMS = ["jev", "no_jev"]


# {{docs-fragment unit}}
@env.task(retries=2)  # deliberately uncached: a benchmark must measure, not replay
async def evaluate_unit(case_id: str, arm: str, repeat: int) -> dict:
    """One case, one arm, one repetition — the atom the matrix is built from.

    `repeat` is part of the signature on purpose: every repetition is a distinct
    action with its own identity, so it is separately retried and separately
    visible in the run graph.
    """
    c = case(case_id)
    s1_in = s1_out = s2_in = s2_out = 0
    latency = 0.0

    if arm == "jev":
        battery, info = await ask_with_info(ReviewBattery, c.state)
        s1_in, s1_out, latency = info.input_tokens, info.output_tokens, info.latency_s
        questions = info.questions
    else:
        async with System2() as s2:
            battery, result = await s2.answer_battery(c.state)
        s2_in, s2_out, latency = result.input_tokens, result.output_tokens, result.latency_s
        questions = len(fired(battery)) + 1

    # Identical from here down in both arms.
    verdict, why = compose(battery)
    tier, gate = route(battery, verdict)

    if tier != "escalate":
        async with System2() as s2:
            review = await s2.write_review(c.state, {"verdict": verdict.value, "why": why, "fired": fired(battery)})
        s2_in += review.input_tokens
        s2_out += review.output_tokens
        latency += review.latency_s

    cost = (
        s1_in / 1e6 * PRICING["system1_input"]
        + s1_out / 1e6 * PRICING["system1_output"]
        + s2_in / 1e6 * PRICING["system2_input"]
        + s2_out / 1e6 * PRICING["system2_output"]
    )
    return {
        "case_id": c.id,
        "arm": arm,
        "repeat": repeat,
        "verdict": verdict.value,
        "truth": c.verdict.value,
        "correct": verdict is c.verdict,
        "route": tier,
        "gate": gate,
        "guard_ok": (not c.hostile) or tier == "escalate",
        "questions": questions,
        "latency_s": round(latency, 3),
        "s1_tokens": s1_in + s1_out,
        "s2_tokens": s2_in + s2_out,
        "cost_usd": cost,
    }
# {{/docs-fragment unit}}


def _summarize(units: list[dict]) -> dict:
    latencies = [u["latency_s"] for u in units]
    return {
        "n": len(units),
        "label_accuracy": sum(u["correct"] for u in units) / len(units),
        "guard_accuracy": sum(u["guard_ok"] for u in units) / len(units),
        "latency_mean": statistics.fmean(latencies),
        "latency_sd": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "cost_per_case": statistics.fmean([u["cost_usd"] for u in units]),
        "s1_tokens": sum(u["s1_tokens"] for u in units),
        "s2_tokens": sum(u["s2_tokens"] for u in units),
        "escalated": sum(u["route"] == "escalate" for u in units),
        "agreement": _modal_agreement(units),
    }


def _modal_agreement(units: list[dict]) -> float:
    """Share of repeats that agree with the modal verdict, averaged over cases.

    The interesting claim about a System One model is not only that it is
    cheaper but that it is *reproducible*: the same diff yields the same typed
    decision run after run, where free-text classification drifts.
    """
    by_case: dict[str, list[str]] = {}
    for u in units:
        by_case.setdefault(u["case_id"], []).append(u["verdict"])
    shares = [max(v.count(x) for x in set(v)) / len(v) for v in by_case.values()]
    return statistics.fmean(shares) if shares else 0.0


# {{docs-fragment benchmark}}
@driver_env.task(report=True)
async def run_benchmark(num_cases: int = 0, repeats: int = 2) -> str:
    """Fan the whole matrix out, then aggregate over the repetitions."""
    cases = CASES[:num_cases] if num_cases else CASES
    units = [
        (c.id, arm, repeat)
        for c in cases
        for arm in ARMS
        for repeat in range(repeats)
    ]

    results = await asyncio.gather(
        *[evaluate_unit(case_id, arm, repeat) for case_id, arm, repeat in units],
        return_exceptions=True,
    )
    ok = [r for r in results if isinstance(r, dict)]
    failed = len(results) - len(ok)

    by_arm = {arm: _summarize([u for u in ok if u["arm"] == arm]) for arm in ARMS if any(u["arm"] == arm for u in ok)}

    def _row(arm: str, s: dict) -> str:
        label = "with System 1" if arm == "jev" else "without"
        return (
            f"<tr><td><b>{label}</b></td><td>{s['n']}</td>"
            f"<td>{s['latency_mean']:.2f}s ± {s['latency_sd']:.2f}</td>"
            f"<td>${s['cost_per_case']:.5f}</td><td>{s['label_accuracy']:.0%}</td>"
            f"<td>{s['guard_accuracy']:.0%}</td><td>{s['agreement']:.0%}</td>"
            f"<td>{s['escalated']}</td><td>{s['s1_tokens']:,}</td><td>{s['s2_tokens']:,}</td></tr>"
        )

    case_rows = "".join(
        "<tr><td>{}</td><td>{}</td>{}</tr>".format(
            c.id,
            c.verdict.value,
            "".join(
                "<td>{}</td>".format(
                    ", ".join(sorted({u["verdict"] for u in ok if u["case_id"] == c.id and u["arm"] == arm})) or "—"
                )
                for arm in ARMS
            ),
        )
        for c in cases
    )

    flyte.report.get_tab("Benchmark").log(
        f"<p>{len(cases)} pull requests × {len(ARMS)} arms × {repeats} repeats = "
        f"<b>{len(units)}</b> independent Flyte actions, {failed} failed. Both arms owe the same "
        "eighteen typed answers and are routed by the same code; only the source of the answers "
        "differs. System 1 tokens are priced at TypeSafe's published input rate (output is charged "
        f"at the input rate — no separate output price is published); System 2 at "
        f"${PRICING['system2_input']}/${PRICING['system2_output']} per MTok.</p>"
        "<table><thead><tr><th>arm</th><th>units</th><th>latency</th><th>$ / case</th>"
        "<th>verdict</th><th>guard</th><th>agreement</th><th>escalated</th>"
        "<th>S1 tokens</th><th>S2 tokens</th></tr></thead>"
        f"<tbody>{''.join(_row(arm, s) for arm, s in by_arm.items())}</tbody></table>"
        "<p><b>Per case.</b> Where a cell holds more than one verdict, the repeats disagreed.</p>"
        "<table><thead><tr><th>case</th><th>truth</th><th>with System 1</th><th>without</th>"
        f"</tr></thead><tbody>{case_rows}</tbody></table>"
    )
    await flyte.report.flush.aio()

    return "; ".join(
        f"{arm}: {s['latency_mean']:.2f}s, ${s['cost_per_case']:.5f}/case, "
        f"verdict {s['label_accuracy']:.0%}, agreement {s['agreement']:.0%}"
        for arm, s in by_arm.items()
    )
# {{/docs-fragment benchmark}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(run_benchmark)
    print(run.name, run.url)
    run.wait()
