# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1",
#    "flyteplugins-union[guardrails]",
# ]
# ///

"""Add guardrail routes to a FastAPI app you already have."""

import pathlib

import flyte
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from flyteplugins.union.guardrails import GuardrailRequest, Verdict, guardrail_router

BANNED_TOPICS = ("competitor pricing", "unreleased roadmap")


def check_prompt(request: GuardrailRequest) -> Verdict:
    prompt = request.last_user_message().lower()
    for topic in BANNED_TOPICS:
        if topic in prompt:
            return Verdict.block(f"Questions about {topic} are not allowed.", reason="banned_topic")
    return Verdict.allow()


# {{docs-fragment router}}
app = FastAPI()


@app.get("/topics")
def topics() -> list[str]:
    return list(BANNED_TOPICS)


app.include_router(guardrail_router(pre=check_prompt))

env = FastAPIAppEnvironment(
    name="topic-guardrail",
    app=app,
    # The gateway only accepts apps of type "Guardrail" as guardrails.
    type="Guardrail",
    image=flyte.Image.from_debian_base().with_pip_packages("flyteplugins-union[guardrails]"),
)
# {{/docs-fragment router}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    print(flyte.deploy(env))
