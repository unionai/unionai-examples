# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
# ]
# main = "summarize"
# params = "text='First document...'"
# ///

"""Call Perplexity from a task with your own API key."""

# {{docs-fragment task}}
import os

import flyte

env = flyte.TaskEnvironment(
    name="perplexity-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("openai"),
    secrets=flyte.Secret(key="perplexity-api-key", as_env_var="PERPLEXITY_API_KEY"),
)


@env.task
async def summarize(text: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://api.perplexity.ai/v1",
        api_key=os.environ["PERPLEXITY_API_KEY"],
    )
    response = await client.responses.create(
        model="<model-id>",
        input=f"Summarize in two sentences:\n\n{text}",
    )
    return response.output_text


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize, text="First document...")
    print(run.url)
# {{/docs-fragment task}}
