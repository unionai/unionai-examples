# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "summarize_batch"
# params = "texts='[\"First document...\", \"Second document...\"]'"
# ///

"""Run a batch on a Fireworks on-demand deployment."""

# {{docs-fragment batch}}
import os
from datetime import timedelta

import flyte

DEPLOYMENT = "accounts/<your-account-id>/deployments/<your-deployment-id>"
REPLICAS = 2
MAX_CONCURRENCY_PER_REPLICA = 8

env = flyte.TaskEnvironment(
    name="fireworks-on-demand",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="fireworks-api-key", as_env_var="FIREWORKS_API_KEY"),
)


# Backs off for up to about 27 minutes while a deployment scales up from zero.
@env.task(
    cache="auto",
    retries=flyte.RetryStrategy(
        count=30,
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
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return response.choices[0].message.content


@env.task
async def summarize_batch(texts: list[str]) -> list[str]:
    summaries = flyte.map.aio(
        summarize,
        texts,
        concurrency=REPLICAS * MAX_CONCURRENCY_PER_REPLICA,
        return_exceptions=False,
    )
    return [summary async for summary in summaries]
# {{/docs-fragment batch}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize_batch, texts=["First document...", "Second document..."])
    print(run.url)
