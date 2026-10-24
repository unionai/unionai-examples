# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1",
#    "openai>=1.60",
# ]
# ///

"""Call models through the Union LLM gateway from a task."""

import os
import pathlib

import flyte

# {{docs-fragment env}}
# The gateway's endpoint, from Settings > LLM Gateway in the Union console.
GATEWAY_URL = "https://<your-gateway-endpoint>"

env = flyte.TaskEnvironment(
    name="llm-gateway-task",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    # Store the virtual key once with: flyte create secret llm-gateway-key sk-un-...
    secrets=flyte.Secret(key="llm-gateway-key", as_env_var="LLM_GATEWAY_KEY"),
)
# {{/docs-fragment env}}


# {{docs-fragment task}}
@env.task
async def summarize(text: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"{GATEWAY_URL}/v1", api_key=os.environ["LLM_GATEWAY_KEY"])
    response = await client.chat.completions.create(
        # Models are addressed as provider/model.
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": f"Summarize in one sentence:\n\n{text}"}],
        # Tried in order if the primary model fails or is rejected for this key.
        extra_body={"fallbacks": ["anthropic/claude-haiku-4-5"]},
    )
    return response.choices[0].message.content
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(summarize, text="Flyte is a workflow orchestrator for data and ML pipelines.")
    print(run.url)
