# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "main"
# params = ""
# ///

import asyncio
from typing import List

import flyte

env = flyte.TaskEnvironment("env")

# {{docs-fragment all}}
@flyte.trace
async def traced_data_cleaning(dataset_id: str) -> List[str]:
    # Creates checkpoint after successful execution.
    await asyncio.sleep(0.2)
    return [f"cleaned_record_{i}_{dataset_id}" for i in range(100)]

@flyte.trace
async def traced_feature_extraction(data: List[str]) -> dict:
    # Creates checkpoint after successful execution.
    await asyncio.sleep(0.3)
    return {
        "features": [f"feature_{i}" for i in range(10)],
        "feature_count": len(data),
        "processed_samples": len(data)
    }

@flyte.trace
async def traced_model_training(features: dict) -> dict:
    # Creates checkpoint after successful execution.
    await asyncio.sleep(0.4)
    sample_count = features["processed_samples"]
    # Mock accuracy based on sample count
    accuracy = min(0.95, 0.7 + (sample_count / 1000))
    return {
        "accuracy": accuracy,
        "epochs": 50,
        "model_size": "125MB"
    }

@env.task(cache="auto")  # Task-level caching enabled
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
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(data_pipeline, "dataset_id_123")
    print(r.name)
    print(r.url)
    r.wait()
