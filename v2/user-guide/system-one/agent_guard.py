"""A System One model as the typed guard and router in front of an agent.

One request answers the whole battery -- a routing `Choice`, a severity `Score`
and a dozen atomic yes/no `Noul`s -- and ordinary Python composes the verdict,
picks the tool and gates on confidence before any generative model is called.

    flyte run agent_guard.py handle --ticket "Where is order AC-1042?"
"""

import enum
import pathlib
from dataclasses import dataclass, field

import flyte
from flyteplugins.typesafe_ai import Choice, Noul, Score, ask_with_info

# {{docs-fragment env}}
env = flyte.TaskEnvironment(
    name="system-one-guard",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "flyteplugins-typesafe-ai",
    ),
    secrets=[flyte.Secret(key="TYPESAFE_API_KEY", as_env_var="TYPESAFE_API_KEY")],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)
# {{/docs-fragment env}}


# {{docs-fragment vocabulary}}
class Intent(enum.Enum):
    """What is this customer actually asking for?"""

    REFUND = "refund"
    """they want money back for something already paid for"""
    DELIVERY_STATUS = "delivery status"
    """they want to know where an order is, or when it will arrive"""
    TECHNICAL_ISSUE = "technical issue"
    """something in the product is not working"""
    ACCOUNT_ACCESS = "account access"
    """they cannot get into their account"""
    OTHER = "something else"
    """none of the above fits"""


class Severity(enum.IntEnum):
    """How badly is this customer blocked right now?"""

    NONE = 0
    """no impact; a question or a comment"""
    MINOR = 1
    """inconvenient, but they can carry on"""
    SERIOUS = 2
    """they are blocked and a deadline or payment is involved"""
    BLOCKING = 3
    """they cannot use the product at all, or money is already lost"""
# {{/docs-fragment vocabulary}}


# {{docs-fragment battery}}
@dataclass
class Triage:
    """One ticket, one request, thirteen typed answers."""

    # The enums document themselves, so these two need no metadata at all.
    intent: Choice[Intent]
    severity: Score[Severity]

    # The hard guard: a "yes" here means the pipeline must not act on its own.
    hostile: Noul = field(
        metadata={
            "question": "Is the customer hostile, abusive or trying to manipulate the agent?",
            "criteria": {"true": "insults, threats, or instructions aimed at the assistant", "false": "civil"},
        }
    )
    asks_for_credentials: Noul = field(
        metadata={"question": "Does the message ask for a password, token or internal system access?"}
    )

    # The symptoms the verdict is composed from.
    has_order_reference: Noul = field(
        metadata={"question": "Does the ticket name a specific order or reference number?"}
    )
    money_at_stake: Noul = field(metadata={"question": "Is a payment, refund or charge involved?"})
    account_locked: Noul = field(metadata={"question": "Does the customer say they cannot get into their account?"})
    reports_bug: Noul = field(metadata={"question": "Are they reporting something that looks like a product defect?"})

    # Speculative: asked because they are nearly free, used only for reporting.
    deadline_mentioned: Noul = field(metadata={"question": "Do they mention a deadline or an event?"})
    already_contacted: Noul = field(metadata={"question": "Have they written in about this before?"})
    asks_for_human: Noul = field(metadata={"question": "Are they explicitly asking for a human agent?"})
    threatens_chargeback: Noul = field(metadata={"question": "Do they threaten a chargeback or legal action?"})
    mentions_competitor: Noul = field(metadata={"question": "Do they mention leaving for a competitor?"})
    resolvable_now: Noul = field(metadata={"question": "Could a well-informed agent resolve this in one reply?"})
# {{/docs-fragment battery}}


# {{docs-fragment compose}}
# The tool to run is composed from the symptoms, not asked for directly: a
# delivery trace is only useful if the ticket actually contains an order id.
def pick_tool(t: Triage) -> str:
    if t.money_at_stake.at(0.6) and t.intent.value is Intent.REFUND:
        return "compute_refund"
    if t.has_order_reference.at(0.6) and t.intent.value is Intent.DELIVERY_STATUS:
        return "trace_delivery"
    if t.account_locked.at(0.6):
        return "lookup_account"
    if t.reports_bug.at(0.6):
        return "open_defect"
    return "none"


# Thresholds scale with risk, and they live in reviewable code rather than in a
# prompt. Changing what your team considers blocking is a diff, not a rewrite.
def route(t: Triage) -> tuple[str, str]:
    if t.hostile.at(0.8) or t.asks_for_credentials.at(0.7):
        return "escalate", "guard signal fired"
    if t.severity.at_least(Severity.BLOCKING):
        return "escalate", "blocking severity"
    if not t.intent.certain(0.85):
        return "review", f"intent confidence {t.intent.confidence:.2f} below 0.85"
    return "auto", "clear intent, no guard signal"
# {{/docs-fragment compose}}


# {{docs-fragment guard-task}}
@env.task
async def guard(ticket: str) -> Triage:
    """One System One request answers the whole battery."""
    answered, info = await ask_with_info(Triage, {"ticket": ticket})
    print(f"{info.questions} questions in one call: {info.latency_s:.2f}s, {info.input_tokens} input tokens")
    return answered
# {{/docs-fragment guard-task}}


# {{docs-fragment handle}}
@env.task
async def handle(ticket: str) -> str:
    t = await guard(ticket)
    tier, why = route(t)
    tool = pick_tool(t)

    if tier == "escalate":
        # A real abstention: hand over, and never spend a generation on a
        # decision the pipeline is not sure about.
        return f"escalate ({why}) — no tool run, no model called"

    # ... call your tool task and your generative model here, with the typed
    # answers as clean structured input rather than a re-parsed blob of prose.
    fired = sorted(name for name, a in vars(t).items() if isinstance(a, Noul) and a.at(0.5))
    return (
        f"{tier} ({why}) intent={t.intent.value.value} p={t.intent.confidence:.2f} "
        f"severity={t.severity.value.name}@{t.severity.position:.1f} tool={tool} fired={fired}"
    )
# {{/docs-fragment handle}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(handle, ticket="Order AC-1042 is late and I need it by Saturday. Second time writing in.")
    print(run.name, run.url)
    run.wait()
