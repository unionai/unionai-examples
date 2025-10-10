# {{docs-fragment auto}}
@env.task(cache=Cache(behavior="auto"))
async def auto_versioned_task(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment auto}}

# {{docs-fragment auto-shorthand}}
@env.task(cache="auto")
async def auto_versioned_task_2(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment auto-shorthand}}

# {{docs-fragment override}}
@env.task(cache=Cache(behavior="override", version_override="v1.2"))
async def manually_versioned_task(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment override}}

# {{docs-fragment disable}}
@env.task(cache=Cache(behavior="disable"))
async def always_fresh_task(data: str) -> str:
    return get_current_timestamp() + data
# {{/docs-fragment disable}}

# {{docs-fragment disable-shorthand}}
@env.task(cache="disable")
async def always_fresh_task_2(x: int) -> int:
    return transform_data(data)
# {{/docs-fragment disable-shorthand}}

# {{docs-fragment ignored}}
@env.task(cache=Cache(
    behavior="auto",
    ignored_inputs=("debug_flag", "logging_level")
))
async def selective_caching(data: str, debug_flag: bool, logging_level: str) -> str:
    if debug_flag:
        print(f"Debug: processing {data}")
    return process_data(data)
# {{/docs-fragment ignored}}

# {{docs-fragment serialize}}
@env.task(cache=Cache(
    behavior="auto",
    serialize=True
))
async def expensive_model_training(dataset: str, params: dict) -> str:
    model = train_large_model(dataset, params)
    return save_model(model)
# {{/docs-fragment serialize}}

# {{docs-fragment salt}}
@env.task(cache=Cache(
    behavior="auto",
    salt="experiment_2024_q4"
))
async def experimental_analysis(data: str) -> dict:
    return run_analysis(data)
# {{/docs-fragment salt}}

# {{docs-fragment policy}}
from flyte import Cache
from flyte._cache import FunctionBodyPolicy

@env.task(cache=Cache(
    behavior="auto",
    policies=[FunctionBodyPolicy()]  # This is the default. Does not actually need to be specified.
))
async def code_sensitive_task(data: str) -> str:
    return data.upper()
# {{/docs-fragment policy}}

# {{docs-fragment custom-policy}}
from flyte._cache import CachePolicy, VersionParameters

class DatasetVersionPolicy(CachePolicy):
    def get_version(self, salt: str, params: VersionParameters) -> str:
        # Generate version based on custom logic
        dataset_version = get_dataset_version()
        return f"{salt}_{dataset_version}"

@env.task(cache=Cache(
    behavior="auto",
    policies=[DatasetVersionPolicy()]
))
async def dataset_dependent_task(data: str) -> str:
    # Cache invalidated when dataset version changes
    return process_with_current_dataset(data)
# {{/docs-fragment custom-policy}}

# {{docs-fragment env-level}}
env = flyte.TaskEnvironment(
    name="cached_environment",
    cache=Cache(behavior="auto")  # Default for all tasks
)

@env.task  # Inherits auto caching from environment
async def inherits_caching(data: str) -> str:
    return process_data(data)
# {{/docs-fragment env-level}}

# {{docs-fragment decorator-level}}
@env.task(cache=Cache(behavior="disable"))  # Override environment default
async def overrides_caching(data: str) -> str:
    return get_timestamp()
# {{/docs-fragment decorator-level}}

# {{docs-fragment override-level}}
@env.task
async def main(data: str) -> str :
    return override_caching_on_call(data).override(cache=Cache(behavior="disable"))
# {{/docs-fragment override-level}}