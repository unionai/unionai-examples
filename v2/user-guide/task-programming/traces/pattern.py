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
async def search_web(query: str) -> list[dict]:
    # Search the web and return results
    await asyncio.sleep(0.1)
    return [{"title": f"Article about {query}", "content": f"Content on {query}"}]

@flyte.trace
async def summarize_content(content: str) -> str:
    # Summarize content using LLM
    await asyncio.sleep(0.1)
    return f"Summary of {len(content.split())} words"

@flyte.trace
async def extract_insights(summaries: list[str]) -> dict:
    # Extract insights from summaries
    await asyncio.sleep(0.1)
    return {"insights": ["key theme 1", "key theme 2"], "count": len(summaries)}

@env.task
async def research_pipeline(topic: str) -> dict:
    # Each helper function creates a checkpoint
    search_results = await search_web(f"research on {topic}")

    summaries = []
    for result in search_results:
        summary = await summarize_content(result["content"])
        summaries.append(summary)

    final_insights = await extract_insights(summaries)

    return {
        "topic": topic,
        "insights": final_insights,
        "sources_count": len(search_results)
    }
# {{/docs-fragment all}}

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(research_pipeline, "machine learning")
    print(r.name)
    print(r.url)
    r.wait()