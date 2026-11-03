# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1",
#    "flyteplugins-union[guardrails]>=0.19.1",
# ]
# ///

"""A guardrail with both stages: block leaked credentials in prompts and in answers."""

import pathlib
import re

import flyte
from flyteplugins.union.guardrails import GuardrailAppEnvironment, GuardrailRequest, Verdict

# {{docs-fragment checks}}
# AWS access key IDs and PEM private keys.
CREDENTIAL = re.compile(r"\bAKIA[0-9A-Z]{16}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")


async def check_prompt(request: GuardrailRequest) -> Verdict:
    if CREDENTIAL.search(request.text()):
        return Verdict.block("Prompts may not contain credentials.", reason="credential_in_prompt")
    return Verdict.allow()


async def check_answer(request: GuardrailRequest) -> Verdict:
    # On the post stage, request.text() is the model's answer.
    if CREDENTIAL.search(request.text()):
        return Verdict.block("The answer contained a credential and was withheld.", reason="credential_in_answer")
    return Verdict.allow()


env = GuardrailAppEnvironment(
    name="credential-guardrail",
    pre=check_prompt,
    post=check_answer,
    scaling=flyte.app.Scaling(replicas=(1, 4)),
)
# {{/docs-fragment checks}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    print(flyte.deploy(env))
