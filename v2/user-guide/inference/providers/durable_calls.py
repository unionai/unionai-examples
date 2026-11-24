# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "summarize_long"
# params = "chunks='[\"First section...\", \"Second section...\"]'"
# ///

"""Keep finished provider calls when a task retries."""

import os

import flyte

env = flyte.TaskEnvironment(
    name="fireworks-durable",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="fireworks-api-key", as_env_var="FIREWORKS_API_KEY"),
)


# {{docs-fragment trace}}
@flyte.trace
async def complete(prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    response = await client.chat.completions.create(
        model="accounts/fireworks/models/glm-5p3-flash",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


@env.task(retries=3)
async def summarize_long(chunks: list[str]) -> str:
    partials = [await complete(f"Summarize in two sentences:\n\n{chunk}") for chunk in chunks]
    return await complete("Combine these summaries into one paragraph:\n\n" + "\n\n".join(partials))
# {{/docs-fragment trace}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize_long, chunks=["First section...", "Second section..."])
    print(run.url)
