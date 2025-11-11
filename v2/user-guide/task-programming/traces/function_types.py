# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "main"
# params = ""
# ///

import asyncio

import flyte

env = flyte.TaskEnvironment("env")

# {{docs-fragment all}}
@flyte.trace
async def async_api_call(topic: str) -> dict:
    # Asynchronous API call
    await asyncio.sleep(0.1)
    return {"data": ["item1", "item2", "item3"], "status": "success"}

@flyte.trace
async def stream_data(items: list[str]):
    # Async generator function for streaming
    for item in items:
        await asyncio.sleep(0.02)
        yield f"Processing: {item}"

@flyte.trace
async def async_stream_llm(prompt: str):
    # Async generator for streaming LLM responses
    chunks = ["Research shows", " that machine learning", " continues to evolve."]
    for chunk in chunks:
        await asyncio.sleep(0.05)
        yield chunk

@env.task
async def research_workflow(topic: str) -> dict:
    llm_result = await async_api_call(topic)

    # Collect async generator results
    processed_data = []
    async for item in stream_data(llm_result["data"]):
        processed_data.append(item)

    llm_stream = []
    async for chunk in async_stream_llm(f"Summarize research on {topic}"):
        llm_stream.append(chunk)

    return {
        "topic": topic,
        "processed_data": processed_data,
        "llm_summary": "".join(llm_stream)
    }
# {{/docs-fragment all}}

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(research_workflow, "machine learning")
    print(r.name)
    print(r.url)
    r.wait()