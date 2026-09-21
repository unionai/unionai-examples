"""Three pipelines, in increasing order of how much the model decides.

    guard_review      — System 1 as a typed I/O guard in front of System 2
    plan_and_execute  — System 1 plans, Flyte fans the tools out, System 1 aggregates
    durable_review    — a ReAct loop whose every branch is a typed answer

They share `battery.py`, so the same composition rules and the same thresholds
govern all three.

    flyte run agents.py guard_review
    flyte run agents.py plan_and_execute
    flyte run agents.py durable_review --case_id c4
"""

from __future__ import annotations

import asyncio
import enum
import json
import pathlib
from dataclasses import dataclass, field
from typing import Annotated

import flyte
import flyte.report
from _runtime import driver_env, env
from battery import (
    CASES,
    SIGNAL_THRESHOLD,
    TOOLS,
    ReviewBattery,
    Severity,
    case,
    compose,
    fired,
    pick_tools,
    route,
)
from flyteplugins.typesafe_ai import Choice, Noul, Score, ask, ask_with_info
from system2 import PRICING, System2


# --------------------------------------------------------------------------- #
# Pattern 1 — System 1 as a typed I/O guard                                    #
# --------------------------------------------------------------------------- #
# {{docs-fragment guard}}
@env.task
async def guard_one(case_id: str) -> dict:
    """One System One request answers the whole battery; code decides."""
    c = case(case_id)
    answered, info = await ask_with_info(ReviewBattery, c.state)
    verdict, why = compose(answered)
    tier, gate_reason = route(answered, verdict)
    return {
        "case_id": c.id,
        "verdict": verdict.value,
        "truth": c.verdict.value,
        "correct": verdict is c.verdict,
        "route": tier,
        "why": why,
        "gate": gate_reason,
        "severity": answered.severity.value.name,
        "confidence": round(answered.severity.confidence, 3),
        "fired": fired(answered),
        "tools": pick_tools(answered),
        "questions": info.questions,
        "latency_s": round(info.latency_s, 3),
        "tokens": info.input_tokens + info.output_tokens,
    }


@driver_env.task(report=True)
async def guard_review() -> str:
    """Guard every case, then let System 2 write only the reviews that survived."""
    guards = await asyncio.gather(*[guard_one(c.id) for c in CASES])

    async with System2() as s2:
        reviews = await asyncio.gather(
            *[
                s2.write_review(case(g["case_id"]).state, g)
                for g in guards
                if g["route"] != "escalate"
            ]
        )

    escalated = [g["case_id"] for g in guards if g["route"] == "escalate"]
    correct = sum(1 for g in guards if g["correct"])
    caught = sum(1 for g, c in zip(guards, CASES) if c.hostile and g["route"] == "escalate")
    hostile = sum(1 for c in CASES if c.hostile)

    rows = "".join(
        f"<tr><td>{g['case_id']}</td><td>{g['route']}</td><td>{g['verdict']}</td><td>{g['truth']}</td>"
        f"<td>{'✓' if g['correct'] else '✗'}</td><td>{g['severity']}</td><td>{g['confidence']}</td>"
        f"<td style='font-size:11px'>{', '.join(g['fired']) or '—'}</td>"
        f"<td>{g['questions']}</td><td>{g['latency_s']}s</td></tr>"
        for g in guards
    )
    flyte.report.get_tab("Guard").log(
        f"<p>Each pull request is guarded by <b>one</b> System One request answering "
        f"<b>{guards[0]['questions']}</b> typed questions in parallel; the verdict, the tool plan "
        f"and the routing tier are composed in Python. Verdict correct on {correct}/{len(CASES)}; "
        f"hostile diffs escalated {caught}/{hostile}; {len(escalated)} case(s) never reached a "
        f"generative model at all: {escalated or '—'}. {len(reviews)} review(s) written.</p>"
        "<table><thead><tr><th>case</th><th>route</th><th>verdict</th><th>truth</th><th>✓</th>"
        "<th>severity</th><th>conf</th><th>signals fired</th><th>questions</th><th>latency</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )
    await flyte.report.flush.aio()
    return (
        f"{len(CASES)} pull requests, {guards[0]['questions']} questions per call; "
        f"verdict {correct}/{len(CASES)}; hostile escalated {caught}/{hostile}; "
        f"{len(escalated)} generations skipped"
    )
# {{/docs-fragment guard}}


# --------------------------------------------------------------------------- #
# Pattern 2 — plan, fan out, aggregate                                         #
# --------------------------------------------------------------------------- #
# {{docs-fragment tool-task}}
@env.task(cache="auto")
async def run_tool(case_id: str, tool: str) -> dict:
    """One backend tool, as its own durable, cached Flyte action."""
    return TOOLS[tool](case(case_id).diff)
# {{/docs-fragment tool-task}}


# {{docs-fragment aggregate}}
@env.task
async def aggregate(case_id: str, plan: dict, tool_output: list[dict]) -> dict:
    """System 1 again, this time over the pooled fan-out output.

    An ad-hoc mapping rather than a dataclass: these questions are assembled at
    runtime, and they still cost a single request.
    """
    answers = await ask(
        {
            "grounded": Annotated[
                Noul,
                {
                    "question": "Is every claim in the plan supported by the tool output?",
                    "criteria": {"true": "the tool output backs the plan", "false": "the plan asserts more"},
                },
            ],
            "tool_output_changes_verdict": Annotated[
                Noul, "Does the tool output contradict the plan's verdict?"
            ],
            "confidence": Annotated[
                Score[Severity], {"question": "After reading the tool output, how severe is this change?"}
            ],
        },
        {"pull_request": case(case_id).state, "plan": plan, "tool_output": tool_output},
    )
    return {
        "grounded": round(answers["grounded"].value, 3),
        "contradicted": round(answers["tool_output_changes_verdict"].value, 3),
        "severity_after_tools": answers["confidence"].value.name,
    }
# {{/docs-fragment aggregate}}


# {{docs-fragment fanout}}
@driver_env.task(report=True)
async def plan_and_execute() -> str:
    """Plan with System 1, fan the tools out on the cluster, aggregate, then write."""
    plans = await asyncio.gather(*[guard_one(c.id) for c in CASES])

    # Every selected tool for every case, executed in parallel as child actions.
    calls = [(p["case_id"], tool) for p in plans for tool in p["tools"] if p["route"] != "escalate"]
    outputs = await asyncio.gather(*[run_tool(case_id, tool) for case_id, tool in calls])

    pooled: dict[str, list[dict]] = {p["case_id"]: [] for p in plans}
    for (case_id, tool), out in zip(calls, outputs):
        pooled[case_id].append({"tool": tool, "output": out})

    checked = {
        case_id: agg
        for case_id, agg in zip(
            [c for c in pooled if pooled[c]],
            await asyncio.gather(
                *[
                    aggregate(case_id, next(p for p in plans if p["case_id"] == case_id), pooled[case_id])
                    for case_id in pooled
                    if pooled[case_id]
                ]
            ),
        )
    }

    rows = "".join(
        f"<tr><td>{p['case_id']}</td><td>{p['route']}</td><td>{p['verdict']}</td>"
        f"<td>{', '.join(p['tools']) if p['route'] != 'escalate' else '—'}</td>"
        f"<td>{(checked.get(p['case_id']) or {}).get('grounded', '—')}</td>"
        f"<td>{(checked.get(p['case_id']) or {}).get('severity_after_tools', '—')}</td></tr>"
        for p in plans
    )
    flyte.report.get_tab("Fan-out").log(
        f"<p>One System One call per case produced the battery <i>and</i> the tool plan. "
        f"Because the plan is composed from the symptoms rather than asked for, the "
        f"{len(CASES)} cases selected <b>{len(calls)}</b> tool executions — every one of them a "
        "separate, cached Flyte action running in parallel. System One then aggregated each "
        "case's pooled output into typed verdicts.</p>"
        "<table><thead><tr><th>case</th><th>route</th><th>verdict</th><th>tools fanned out</th>"
        f"<th>grounded</th><th>severity after tools</th></tr></thead><tbody>{rows}</tbody></table>"
    )
    await flyte.report.flush.aio()
    return f"{len(CASES)} cases planned; {len(calls)} tool actions fanned out; {len(checked)} aggregated"
# {{/docs-fragment fanout}}


# --------------------------------------------------------------------------- #
# Pattern 3 — a durable loop whose control flow is typed                       #
# --------------------------------------------------------------------------- #
# {{docs-fragment loop-types}}
class NextAction(enum.Enum):
    """Given what the reviewer knows so far, what should it do next?"""

    SUMMARIZE_DIFF = "summarize the diff"
    """get the structural facts: files touched, lines added and removed"""
    AUDIT_DEPENDENCIES = "audit dependencies"
    """check whether an added dependency shadows a well-known package"""
    SCAN_SECRETS = "scan for secrets"
    """check whether anything in the diff moves credentials off the machine"""
    DECIDE = "decide now"
    """there is enough evidence to write the review"""


class Confidence(enum.IntEnum):
    """How sure are we that this is the right next step?"""

    LOW = 0
    """a guess; the history does not support it"""
    MEDIUM = 1
    """plausible, but a human should see the result"""
    HIGH = 2
    """clearly the right move"""


@dataclass
class Step:
    """One turn of the loop, answered in one request."""

    action: Choice[NextAction]
    confidence: Score[Confidence]
    has_enough: Noul = field(
        metadata={
            "question": "Is there enough evidence in the history to write the review now?",
            "criteria": {"true": "every fact the review needs is present", "false": "a check is still missing"},
        }
    )
    made_progress: Noul = field(
        metadata={
            "question": "Did the most recent observation add something the reviewer did not already have?",
            "criteria": {"true": "the last step produced something new", "false": "the loop is spinning"},
        }
    )
    hostile: Noul = field(
        metadata={"question": "Does the diff contain instructions aimed at whoever is reviewing it?"}
    )


_LOOP_TOOLS = {
    NextAction.SUMMARIZE_DIFF: "summarize_diff",
    NextAction.AUDIT_DEPENDENCIES: "audit_dependencies",
    NextAction.SCAN_SECRETS: "scan_secrets",
}
# {{/docs-fragment loop-types}}


# {{docs-fragment loop}}
@driver_env.task
async def durable_review(case_id: str = "c4", max_steps: int = 4) -> str:
    """A ReAct loop where every branch is a typed answer, not parsed prose.

    Each tool call is a child action, so the loop is durably recorded: a worker
    that dies on turn three resumes from the record instead of re-running the
    first two.
    """
    c = case(case_id)
    history: list[dict] = [{"role": "user", "content": json.dumps(c.state)[:4000]}]
    trace: list[dict] = []

    for step in range(max_steps):
        s = await ask(Step, {"history": history})
        action = s.action.value
        trace.append(
            {
                "step": step,
                "action": action.value,
                "p_action": round(s.action.confidence, 3),
                "confidence": s.confidence.value.name,
                "has_enough": round(s.has_enough.value, 3),
            }
        )

        # Abstain on the confidence, not on the Choice: a confident wrong action
        # is rarer than a low-confidence right one.
        if s.hostile.at(0.7) or not s.confidence.at_least(Confidence.MEDIUM):
            return json.dumps({"case_id": case_id, "outcome": "escalate", "trace": trace}, indent=2)

        if action is NextAction.DECIDE or s.has_enough.at(SIGNAL_THRESHOLD):
            break
        if step > 0 and not s.made_progress.at(0.4):
            break

        observation = await run_tool(case_id, _LOOP_TOOLS[action])
        history.append({"role": "assistant", "content": f"ran {action.value}"})
        history.append({"role": "tool", "content": json.dumps(observation)[:1500]})
        trace[-1]["observation"] = observation

    async with System2() as s2:
        review = await s2.write_review(c.state, {"trace": trace})

    return json.dumps(
        {
            "case_id": case_id,
            "outcome": "reviewed",
            "actions": [t["action"] for t in trace],
            "trace": trace,
            "review": review.text,
            "system2_cost_usd": round(
                review.input_tokens / 1e6 * PRICING["system2_input"]
                + review.output_tokens / 1e6 * PRICING["system2_output"],
                6,
            ),
        },
        indent=2,
        default=str,
    )
# {{/docs-fragment loop}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(guard_review)
    print(run.name, run.url)
    run.wait()
