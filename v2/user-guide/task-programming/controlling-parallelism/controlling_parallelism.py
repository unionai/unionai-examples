# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "process_batch_with_semaphore"
# params = "prompts = ['Summarize this text', 'Translate to French', 'Extract key points', 'Generate a title', 'Write a conclusion', 'List the main topics', 'Identify the tone', 'Suggest improvements']"
# ///

# {{docs-fragment setup}}
import asyncio

import flyte

env = flyte.TaskEnvironment("controlling_parallelism")


@env.task
async def call_llm_api(prompt: str) -> str:
    """Simulate calling a rate-limited LLM API."""
    # In a real workflow, this would call an external API.
    # The API might allow only a few concurrent requests.
    await asyncio.sleep(0.5)
    return f"Response to: {prompt}"
# {{/docs-fragment setup}}


# {{docs-fragment unbounded}}
@env.task
async def process_all_at_once(prompts: list[str]) -> list[str]:
    """Send all requests in parallel with no concurrency limit.

    This can overwhelm a rate-limited API, causing errors or throttling.
    """
    results = await asyncio.gather(*[call_llm_api(p) for p in prompts])
    return list(results)
# {{/docs-fragment unbounded}}


# {{docs-fragment semaphore}}
@env.task
async def process_batch_with_semaphore(
    prompts: list[str],
    max_concurrent: int = 3,
) -> list[str]:
    """Process prompts in parallel, limiting concurrency with a semaphore.

    At most `max_concurrent` calls to the API run at any given time.
    The remaining tasks wait until a slot is available.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(prompt: str) -> str:
        async with semaphore:
            return await call_llm_api(prompt)

    results = await asyncio.gather(*[limited_call(p) for p in prompts])
    return list(results)
# {{/docs-fragment semaphore}}


# {{docs-fragment map-concurrency}}
@env.task
async def process_batch_with_map(prompts: list[str]) -> list[str]:
    """Process prompts using flyte.map with a built-in concurrency limit.

    This is the simplest approach when every item goes through the same task.
    """
    results = list(flyte.map(call_llm_api, prompts, concurrency=3))
    return results
# {{/docs-fragment map-concurrency}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    prompts = [
        "Summarize this text",
        "Translate to French",
        "Extract key points",
        "Generate a title",
        "Write a conclusion",
        "List the main topics",
        "Identify the tone",
        "Suggest improvements",
    ]
    r = flyte.run(process_batch_with_semaphore, prompts)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}
