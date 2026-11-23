# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "main"
# params = "texts='[\"First document...\", \"Second document...\"]'"
# ///

"""Retry, cache, and limit concurrency for provider calls."""

# {{docs-fragment task}}
import os
from datetime import timedelta

import flyte

env = flyte.TaskEnvironment(
    name="fireworks-reliable",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="fireworks-api-key", as_env_var="FIREWORKS_API_KEY"),
)


@env.task(
    cache="auto",
    retries=flyte.RetryStrategy(
        count=10,
        backoff=flyte.Backoff(base=timedelta(seconds=5), factor=2.0, cap=timedelta(seconds=60)),
    ),
)
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
    summaries = flyte.map.aio(summarize, texts, concurrency=8, return_exceptions=False)
    return [summary async for summary in summaries]
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, texts=["First document...", "Second document..."])
    print(run.url)
