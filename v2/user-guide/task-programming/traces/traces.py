# {{docs-fragment tasks-vs-traced}}
import flyte

env = flyte.TaskEnvironment("my-app")

# Traced helper functions - the main use case
@flyte.trace
async def call_llm(prompt: str) -> str:
    """Helper function for LLM calls - traced for observability."""
    response = await llm_client.chat(prompt)
    return response

@flyte.trace
async def process_data(data: str) -> dict:
    """Helper function for data processing - traced for checkpointing."""
    return await expensive_computation(data)

# Tasks orchestrate traced helper functions
@env.task
async def research_workflow(topic: str) -> dict:
    # Task coordinates the workflow
    llm_result = await call_llm(f"Generate research plan for: {topic}")
    processed_data = await process_data(llm_result)

    return {"topic": topic, "result": processed_data}
# {{/docs-fragment tasks-vs-traced}}

# {{docs-fragment context}}
@flyte.trace
def sync_function(x: int) -> int:
    return x * 2

@flyte.trace
async def async_function(x: int) -> int:
    return x * 2

# ❌ Outside task context - neither work
sync_function(5)  # Fails
await async_function(5)  # Fails

# ✅ Within task context - both work and are traced
@env.task
async def my_task():
    result1 = sync_function(5)       # ✅ Traced! (Coming soon)
    result2 = await async_function(5) # ✅ Traced!
    return result1 + result2
# {{/docs-fragment context}}

# {{docs-fragment function-types}}
@flyte.trace
def sync_process(data: str) -> str:
    """Synchronous data processing."""
    return data.upper()

@flyte.trace
async def async_api_call(url: str) -> dict:
    """Asynchronous API call."""
    response = await http_client.get(url)
    return response.json()

@flyte.trace
def stream_data(items: list[str]):
    """Generator function for streaming."""
    for item in items:
        yield f"Processing: {item}"

@flyte.trace
async def async_stream_llm(prompt: str):
    """Async generator for streaming LLM responses."""
    async for chunk in llm_client.stream(prompt):
        yield chunk
# {{/docs-fragment function-types}}

# {{docs-fragment pattern}}
# Helper functions - traced for observability
@flyte.trace
async def search_web(query: str) -> list[dict]:
    """Search the web and return results."""
    results = await search_api.query(query)
    return results

@flyte.trace
async def summarize_content(content: str) -> str:
    """Summarize content using LLM."""
    summary = await llm_client.summarize(content)
    return summary

@flyte.trace
async def extract_insights(summaries: list[str]) -> dict:
    """Extract insights from summaries."""
    insights = await analysis_service.extract_insights(summaries)
    return insights

# Task - orchestrates the traced helper functions
@env.task
async def research_pipeline(topic: str) -> dict:
    """Main research pipeline task."""

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
# {{/docs-fragment pattern}}

# {{docs-fragment caching-vs-checkpointing}}
@env.task  # Task-level caching enabled by default
async def data_pipeline(dataset_id: str) -> dict:
    # 1. If this exact task with these inputs ran before,
    #    the entire task result is returned from cache

    # 2. If not cached, execution begins and each traced function
    #    creates checkpoints for resumption
    cleaned_data = await traced_data_cleaning(dataset_id)      # Checkpoint 1
    features = await traced_feature_extraction(cleaned_data)   # Checkpoint 2
    model_results = await traced_model_training(features)      # Checkpoint 3

    # 3. If workflow fails at step 3, resumption will:
    #    - Skip traced_data_cleaning (checkpointed)
    #    - Skip traced_feature_extraction (checkpointed)
    #    - Re-run only traced_model_training

    return {"dataset_id": dataset_id, "accuracy": model_results["accuracy"]}

@flyte.trace
async def traced_data_cleaning(dataset_id: str) -> list:
    """Creates checkpoint after successful execution."""
    return await expensive_cleaning_process(dataset_id)

@flyte.trace
async def traced_feature_extraction(data: list) -> dict:
    """Creates checkpoint after successful execution."""
    return await expensive_feature_process(data)

@flyte.trace
async def traced_model_training(features: dict) -> dict:
    """Creates checkpoint after successful execution."""
    return await expensive_training_process(features)
# {{/docs-fragment caching-vs-checkpointing}}

# {{docs-fragment error-handling}}
@flyte.trace
async def risky_api_call(endpoint: str, data: dict) -> dict:
    """API call that might fail - traces capture errors."""
    try:
        response = await api_client.post(endpoint, json=data)
        return response.json()
    except Exception as e:
        # Error is automatically captured in trace
        logger.error(f"API call failed: {e}")
        raise

@env.task
async def error_handling_workflow():
    try:
        result = await risky_api_call("/process", {"invalid": "data"})
        return {"status": "success", "result": result}
    except Exception as e:
        # The error is recorded in the trace for debugging
        return {"status": "error", "message": str(e)}
# {{/docs-fragment error-handling}}

# {{docs-fragment recommended}}
# ✅ Traced helper functions for specific operations
@flyte.trace
async def call_llm(prompt: str, model: str) -> str:
    """LLM interaction - traced for observability."""
    return await llm_client.chat(prompt, model=model)

@flyte.trace
async def search_database(query: str) -> list[dict]:
    """Database operation - traced for checkpointing."""
    return await db.search(query)

@flyte.trace
async def process_results(data: list[dict]) -> dict:
    """Data processing - traced for error tracking."""
    return await expensive_analysis(data)

# ✅ Tasks that orchestrate traced functions
@env.task
async def research_workflow(topic: str) -> dict:
    """Main workflow - coordinates traced operations."""
    search_results = await search_database(f"research: {topic}")
    analysis_prompt = f"Analyze these results: {search_results}"
    llm_analysis = await call_llm(analysis_prompt, "gpt-4")
    final_results = await process_results([{"analysis": llm_analysis}])

    return final_results
# {{/docs-fragment recommended}}

