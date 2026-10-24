# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1",
#    "flyteplugins-union[guardrails]",
# ]
# ///

"""A pre-call guardrail that blocks personal data and answers known questions from a cache."""

import pathlib
import re

import flyte
from flyteplugins.union.guardrails import GuardrailAppEnvironment, GuardrailRequest, Verdict

# {{docs-fragment checks}}
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

# Questions answered without calling a model.
CACHE = {
    "what is our refund window?": "Refunds are accepted within 30 days of purchase.",
}


def check_prompt(request: GuardrailRequest) -> Verdict:
    prompt = request.last_user_message()
    if SSN.search(prompt):
        return Verdict.block("Remove the Social Security number and try again.", reason="pii_ssn")
    if EMAIL.search(prompt):
        return Verdict.block("Remove the email address and try again.", reason="pii_email")
    if answer := CACHE.get(prompt.strip().lower()):
        return Verdict.cached(answer)
    return Verdict.allow()
# {{/docs-fragment checks}}


# {{docs-fragment env}}
env = GuardrailAppEnvironment(
    name="pii-guardrail",
    pre=check_prompt,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    # Keep a replica warm: every guarded call waits on this app.
    scaling=flyte.app.Scaling(replicas=(1, 4)),
)
# {{/docs-fragment env}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployed = flyte.deploy(env)
    print(deployed)
