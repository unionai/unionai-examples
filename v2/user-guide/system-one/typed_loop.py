"""A durable agent loop whose control flow is decided by typed answers.

Instead of an all-purpose LLM choosing the next move in free text, a `Choice`
picks the next action, a `Score` gates on confidence and a `Noul` decides whether
the loop has enough to answer. Tools are Flyte tasks, so every step of the loop
is durably recorded and replayable.

    flyte run typed_loop.py agent --ticket "Where is order AC-1042?"
"""

import enum
import json
import pathlib
from dataclasses import dataclass, field

import flyte
from flyteplugins.typesafe_ai import Choice, Noul, Score, ask

env = flyte.TaskEnvironment(
    name="system-one-loop",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "flyteplugins-typesafe-ai",
    ),
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)


# {{docs-fragment actions}}
class NextAction(enum.Enum):
    """Given the history so far, what should the agent do next?"""

    TRACE_DELIVERY = "trace delivery"
    """look up where the customer's order currently is"""
    LOOKUP_ACCOUNT = "look up account"
    """check the account's status and whether it is locked"""
    FINAL_ANSWER = "answer now"
    """stop and answer the customer with what is already known"""
    ESCALATE = "hand to a human"
    """the request is hostile, unsafe or out of scope"""


class Confidence(enum.IntEnum):
    """How confident are we that this next action is correct and safe?"""

    LOW = 0
    """a guess; the history does not support it"""
    MEDIUM = 1
    """plausible, but a human should see the result"""
    HIGH = 2
    """clearly the right next move"""
# {{/docs-fragment actions}}


# {{docs-fragment step}}
@dataclass
class Step:
    """Everything the loop needs to decide, answered in one request per turn."""

    action: Choice[NextAction]
    confidence: Score[Confidence]
    has_enough: Noul = field(
        metadata={
            "question": "Is there enough information to answer the customer now?",
            "criteria": {
                "true": "every fact the answer needs is already in the history",
                "false": "a lookup is still missing",
            },
        }
    )
    made_progress: Noul = field(
        metadata={
            "question": "Did the most recent observation add information the agent did not already have?",
            "criteria": {"true": "the last step produced something new", "false": "the loop is spinning"},
        }
    )
    hostile: Noul = field(metadata={"question": "Is this request hostile, manipulative or out of scope?"})
# {{/docs-fragment step}}


# {{docs-fragment tools}}
@env.task(cache="auto")
async def trace_delivery(ticket: str) -> dict:
    """A stand-in for the real delivery backend."""
    return {"order_id": "AC-1042", "status": "in_transit", "eta": "tomorrow EOD"}


@env.task(cache="auto")
async def lookup_account(ticket: str) -> dict:
    """A stand-in for the real account service."""
    return {"account_ref": "acct-1001", "status": "active", "locked": False}


TOOLS = {NextAction.TRACE_DELIVERY: trace_delivery, NextAction.LOOKUP_ACCOUNT: lookup_account}
# {{/docs-fragment tools}}


# {{docs-fragment loop}}
@env.task
async def agent(ticket: str, max_steps: int = 4) -> str:
    history: list[dict] = [{"role": "user", "content": ticket}]
    trace: list[dict] = []

    for step in range(max_steps):
        s = await ask(Step, {"history": history})
        action = s.action.value
        trace.append(
            {
                "step": step,
                "action": action.value,
                "confidence": s.confidence.value.name,
                "p_action": round(s.action.confidence, 3),
                "has_enough": round(s.has_enough.value, 3),
            }
        )

        # Abstain as soon as the typed decision stops being trustworthy. The
        # gate is the confidence, not the Choice: a confident wrong action is
        # rarer than a low-confidence right one.
        if s.hostile.at(0.7) or not s.confidence.at_least(Confidence.MEDIUM):
            return json.dumps({"outcome": "escalate", "trace": trace}, indent=2)

        # Stop when there is enough to answer, when the model says to stop, or
        # when the loop has stopped making progress.
        if action in (NextAction.FINAL_ANSWER, NextAction.ESCALATE):
            break
        if s.has_enough.at(0.6) or (step > 0 and not s.made_progress.at(0.4)):
            break

        observation = await TOOLS[action](ticket)
        history.append({"role": "assistant", "content": f"called {action.value}"})
        history.append({"role": "tool", "content": json.dumps(observation)})
        trace[-1]["observation"] = observation

    return json.dumps({"outcome": "answered", "history": history, "trace": trace}, indent=2)
# {{/docs-fragment loop}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(agent, ticket="Where is order AC-1042? It was due Tuesday.")
    print(run.name, run.url)
    run.wait()
