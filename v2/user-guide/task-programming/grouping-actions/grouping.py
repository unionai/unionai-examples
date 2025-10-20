# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = ""
# ///

import flyte
import asyncio
from typing import List

env = flyte.TaskEnvironment(name="grouping-actions")

# Reusable sub-tasks that can be used across different examples
@env.task
async def process_data(data: str, step_name: str = "process") -> str:
    """Generic data processing task - can represent validation, transformation, etc."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"{step_name}({data})"

@env.task
async def process_item(item: int, operation: str = "transform") -> int:
    """Generic item processing - can represent preprocessing, training, etc."""
    await asyncio.sleep(0.05)  # Simulate work
    return item * 2 if operation == "transform" else item + 10

@env.task
async def aggregate_results(results: List[str], final_step: str = "aggregate") -> str:
    """Combine multiple results into a final output."""
    combined = "+".join(results)
    return f"{final_step}({combined})"

# {{docs-fragment sequential}}
@env.task
async def data_pipeline(raw_data: str) -> str:
    with flyte.group("data-validation"):
        validated_data = await process_data(raw_data, "validate_schema")
        validated_data = await process_data(validated_data, "check_quality")
        validated_data = await process_data(validated_data, "remove_duplicates")

    with flyte.group("feature-engineering"):
        features = await process_data(validated_data, "extract_features")
        features = await process_data(features, "normalize_features")
        features = await process_data(features, "select_features")

    with flyte.group("model-training"):
        model = await process_data(features, "train_model")
        model = await process_data(model, "validate_model")
        final_model = await process_data(model, "save_model")

    return final_model
# {{/docs-fragment sequential}}

# {{docs-fragment parallel}}
@env.task
async def parallel_processing_example(n: int) -> str:
    tasks = []

    with flyte.group("parallel-processing"):
        # Collect all task invocations first
        for i in range(n):
            tasks.append(process_item(i, "transform"))

        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks)

    # Convert to string for consistent return type
    return f"parallel_results: {results}"
# {{/docs-fragment parallel}}

# {{docs-fragment multi}}
@env.task
async def multi_phase_workflow(data_size: int) -> str:
    # First phase: data preprocessing
    preprocessed = []
    with flyte.group("preprocessing"):
        for i in range(data_size):
            preprocessed.append(process_item(i, "preprocess"))
        phase1_results = await asyncio.gather(*preprocessed)

    # Second phase: main processing
    processed = []
    with flyte.group("main-processing"):
        for result in phase1_results:
            processed.append(process_item(result, "transform"))
        phase2_results = await asyncio.gather(*processed)

    # Third phase: postprocessing
    postprocessed = []
    with flyte.group("postprocessing"):
        for result in phase2_results:
            postprocessed.append(process_item(result, "postprocess"))
        final_results = await asyncio.gather(*postprocessed)

    # Convert to string for consistent return type
    return f"multi_phase_results: {final_results}"
# {{/docs-fragment multi}}

# {{docs-fragment nested}}
@env.task
async def hierarchical_example(raw_data: str) -> str:
    with flyte.group("machine-learning-pipeline"):
        with flyte.group("data-preparation"):
            cleaned_data = await process_data(raw_data, "clean_data")
            split_data = await process_data(cleaned_data, "split_dataset")

        with flyte.group("model-experiments"):
            with flyte.group("hyperparameter-tuning"):
                best_params = await process_data(split_data, "tune_hyperparameters")

            with flyte.group("model-training"):
                model = await process_data(best_params, "train_final_model")
    return model
# {{/docs-fragment nested}}

# {{docs-fragment conditional}}
@env.task
async def conditional_processing(use_advanced_features: bool, input_data: str) -> str:
    base_result = await process_data(input_data, "basic_processing")

    if use_advanced_features:
        with flyte.group("advanced-features"):
            enhanced_result = await process_data(base_result, "advanced_processing")
            optimized_result = await process_data(enhanced_result, "optimize_result")
            return optimized_result
    else:
        with flyte.group("basic-features"):
            simple_result = await process_data(base_result, "simple_processing")
            return simple_result
# {{/docs-fragment conditional}}


# {{docs-fragment run}}
@env.task
async def main() -> List[str]:
    # Run all example workflows - all return strings now
    grouped_results = await data_pipeline("raw_input_data")
    parallel_results = await parallel_processing_example(5)
    multi_phase_results = await multi_phase_workflow(3)
    hierarchical_result = await hierarchical_example("input_data")
    conditional_result = await conditional_processing(True, "test_data")

    return [grouped_results, parallel_results, multi_phase_results, hierarchical_result, conditional_result]


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}