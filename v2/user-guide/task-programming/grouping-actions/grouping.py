# {{docs-fragment simple}}
with flyte.group("my-group-name"):
    # All task invocations here belong to "my-group-name"
    result1 = await task_a(data)
    result2 = await task_b(data)
    result3 = await task_c(data)
# {{/docs-fragment simple}}

# {{docs-fragment sequential}}
@env.task
async def data_pipeline(raw_data: str) -> str:
    with flyte.group("data-validation"):
        validated_data = await validate_schema(raw_data)
        validated_data = await check_data_quality(validated_data)
        validated_data = await remove_duplicates(validated_data)

    with flyte.group("feature-engineering"):
        features = await extract_features(validated_data)
        features = await normalize_features(features)
        features = await select_features(features)

    with flyte.group("model-training"):
        model = await train_model(features)
        model = await validate_model(model)
        final_model = await save_model(model)

    return final_model
# {{/docs-fragment sequential}}

# {{docs-fragment parallel}}
async def parallel_processing_example(n: int) -> List[str]:
    results = []

    with flyte.group("parallel-processing"):
        # Collect all task invocations first
        for i in range(n):
            results.append(my_async_task(i))

        # Execute all tasks in parallel
        final_results = await asyncio.gather(*results)

    return final_results
# {{/docs-fragment parallel}}

# {{docs-fragment multi}}
@env.task
async def multi_phase_workflow(data_size: int) -> List[int]:
    # First phase: data preprocessing
    preprocessed = []
    with flyte.group("preprocessing"):
        for i in range(data_size):
            preprocessed.append(preprocess_task(i))
        phase1_results = await asyncio.gather(*preprocessed)

    # Second phase: main processing
    processed = []
    with flyte.group("main-processing"):
        for result in phase1_results:
            processed.append(process_task(result))
        phase2_results = await asyncio.gather(*processed)

    # Third phase: postprocessing
    postprocessed = []
    with flyte.group("postprocessing"):
        for result in phase2_results:
            postprocessed.append(postprocess_task(result))
        final_results = await asyncio.gather(*postprocessed)

    return final_results
# {{/docs-fragment multi}}

# {{docs-fragment nested}}
@env.task
async def hierarchical_example():
    with flyte.group("machine-learning-pipeline"):
        with flyte.group("data-preparation"):
            cleaned_data = await clean_data(raw_data)
            split_data = await split_dataset(cleaned_data)

        with flyte.group("model-experiments"):
            with flyte.group("hyperparameter-tuning"):
                best_params = await tune_hyperparameters(split_data)

            with flyte.group("model-training"):
                model = await train_final_model(split_data, best_params)
# {{/docs-fragment nested}}

# {{docs-fragment conditional}}
@env.task
async def conditional_processing(use_advanced_features: bool):
    base_result = await basic_processing()

    if use_advanced_features:
        with flyte.group("advanced-features"):
            enhanced_result = await advanced_processing(base_result)
            optimized_result = await optimize_result(enhanced_result)
            return optimized_result
    else:
        with flyte.group("basic-features"):
            simple_result = await simple_processing(base_result)
            return simple_result
# {{/docs-fragment conditional}}

# {{docs-fragment meaningful}}
with flyte.group("feature-extraction"):
with flyte.group("model-training"):
with flyte.group("hyperparameter-sweep"):
# {{/docs-fragment meaningful}}

# {{docs-fragment logical}}

# Good: Group by logical phase
with flyte.group("data-validation"):
    # All validation tasks
with flyte.group("feature-engineering"):
    # All feature engineering tasks
# {{/docs-fragment logical}}

# {{docs-fragment consistent}}
# Use consistent prefixes or patterns
with flyte.group("phase-1-preprocessing"):
with flyte.group("phase-2-training"):
with flyte.group("phase-3-evaluation"):
# {{/docs-fragment consistent}}

# {{docs-fragment granularity}}
# Too granular - avoid
with flyte.group("step-1"):
    task_a()
with flyte.group("step-2"):
    task_b()

# Better - logical grouping
with flyte.group("data-preparation"):
    task_a()
    task_b()
    task_c()
# {{/docs-fragment granularity}}
