# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "anthropic>=0.40",
# ]
# main = "summarize"
# params = "text='First document...'"
# ///

"""Call Anthropic from a task with your own API key."""

# {{docs-fragment task}}
import os

import flyte

env = flyte.TaskEnvironment(
    name="anthropic-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("anthropic"),
    secrets=flyte.Secret(key="anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


@env.task
async def summarize(text: str) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = await client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return response.content[0].text


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize, text="First document...")
    print(run.url)
# {{/docs-fragment task}}
