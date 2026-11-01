# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# ///

"""Use a different provider API key for one task invocation."""

import os

import flyte

env = flyte.TaskEnvironment(
    name="fireworks-per-team",
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
async def summarize_for_team_b(text: str) -> str:
    # {{docs-fragment override}}
    team_b = summarize.override(
        secrets=flyte.Secret(key="fireworks-api-key-team-b", as_env_var="FIREWORKS_API_KEY")
    )
    result = await team_b(text=text)
    # {{/docs-fragment override}}
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize_for_team_b, text="First document...")
    print(run.url)
