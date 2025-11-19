# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "research_workflow"
# params = 'topic="machine learning"'
# ///

# {{docs-fragment all}}
import asyncio

import flyte

env = flyte.TaskEnvironment("env")


@flyte.trace
async def call_llm(prompt: str) -> str:
    await asyncio.sleep(0.1)
    return f"LLM response for: {prompt}"


@flyte.trace
async def process_data(data: str) -> dict:
    await asyncio.sleep(0.2)
    return {"processed": data, "status": "completed"}


@env.task
async def research_workflow(topic: str) -> dict:
    llm_result = await call_llm(f"Generate research plan for: {topic}")
    processed_data = await process_data(llm_result)
    return {"topic": topic, "result": processed_data}
# {{/docs-fragment all}}

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(research_workflow, "machine learning")
    print(r.name)
    print(r.url)
    r.wait()