# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1",
#    "openai>=1.60",
#    "fastapi",
#    "uvicorn",
# ]
# ///

"""Call models through the Union LLM gateway from an app."""

import os
import pathlib

import flyte
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment

GATEWAY_URL = "https://<your-gateway-endpoint>"

# {{docs-fragment app}}
app = FastAPI()


@app.get("/ask")
async def ask(question: str) -> dict[str, str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"{GATEWAY_URL}/v1",
        api_key=os.environ["LLM_GATEWAY_KEY"],
        # Tag requests so the console can break down spend by app.
        default_headers={"x-un-dim-app": "support-bot"},
    )
    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": question}],
    )
    return {"answer": response.choices[0].message.content}


env = FastAPIAppEnvironment(
    name="support-bot",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "openai"),
    # Each app gets its own virtual key, so its budget and allowed models are its own.
    secrets=flyte.Secret(key="support-bot-llm-key", as_env_var="LLM_GATEWAY_KEY"),
)
# {{/docs-fragment app}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployed = flyte.serve(env)
    print(deployed.url)
