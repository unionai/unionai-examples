# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.12.1",
# ]
# main = "pipeline"
# params = "x=3"
# ///

# {{docs-fragment notify-from-code}}
import os

import httpx

import flyte

env = flyte.TaskEnvironment(
    name="notify_from_code",
    secrets=[flyte.Secret(key="SLACK_WEBHOOK_URL")],
)


@flyte.trace
async def post_slack(text: str) -> None:
    # The secret is read when the task runs, not when it is deployed.
    async with httpx.AsyncClient() as client:
        response = await client.post(os.environ["SLACK_WEBHOOK_URL"], json={"text": text})
        response.raise_for_status()


@env.task(retries=3)
async def train(x: int) -> int:
    if x < 0:
        raise ValueError("x must be non-negative")
    return x * 2


@env.task
async def pipeline(x: int) -> int:
    run_name = flyte.ctx().action.run_name
    try:
        result = await train(x)
    except Exception as e:
        await post_slack(f"Run {run_name} failed: {e}")
        raise
    await post_slack(f"Run {run_name} succeeded with result {result}.")
    return result
# {{/docs-fragment notify-from-code}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline, x=3)
    print(run.url)
