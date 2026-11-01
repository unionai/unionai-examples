# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "main"
# params = "texts='[\"First document...\", \"Second document...\"]'"
# ///

"""Call Fireworks AI from a task with your own API key."""

# {{docs-fragment task}}
import asyncio
import os

import flyte

env = flyte.TaskEnvironment(
    name="fireworks-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="fireworks-api-key", as_env_var="FIREWORKS_API_KEY"),
)


@env.task
async def summarize(text: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    response = await client.chat.completions.create(
        model="accounts/fireworks/models/glm-5p3-flash",
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return response.choices[0].message.content


@env.task
async def main(texts: list[str]) -> list[str]:
    return await asyncio.gather(*(summarize(text) for text in texts))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, texts=["First document...", "Second document..."])
    print(run.url)
# {{/docs-fragment task}}

# {{docs-fragment on-demand}}
# An on-demand deployment, in place of the model name above.
on_demand_model = "accounts/<your-account-id>/deployments/<your-deployment-id>"
# {{/docs-fragment on-demand}}
