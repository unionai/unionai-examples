# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "summarize"
# params = "text='First document...'"
# ///

"""Call Sarvam AI from a task with your own API key."""

# {{docs-fragment task}}
import os

import flyte

env = flyte.TaskEnvironment(
    name="sarvam-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="sarvam-api-key", as_env_var="SARVAM_API_KEY"),
)


@env.task
async def summarize(text: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://api.sarvam.ai/v2",
        api_key=os.environ["SARVAM_API_KEY"],
    )
    response = await client.chat.completions.create(
        model="<model-id>",
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize, text="First document...")
    print(run.url)
# {{/docs-fragment task}}
