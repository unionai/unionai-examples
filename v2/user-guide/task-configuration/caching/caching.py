# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = ""
# ///

import flyte

env = flyte.TaskEnvironment(name="env")


async def transform_data(data: str) -> str:
    return data.upper()


# {{docs-fragment auto}}
@env.task(cache=flyte.Cache(behavior="auto"))
async def auto_versioned_task(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment auto}}


# {{docs-fragment auto-shorthand}}
@env.task(cache="auto")
async def auto_versioned_task_2(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment auto-shorthand}}


# {{docs-fragment override}}
@env.task(cache=flyte.Cache(behavior="override", version_override="v1.2"))
async def manually_versioned_task(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment override}}


# {{docs-fragment disable}}
@env.task(cache=flyte.Cache(behavior="disable"))
async def always_fresh_task(data: str) -> str:
    return get_current_timestamp() + transform_data(data)
# {{/docs-fragment disable}}


# {{docs-fragment disable-shorthand}}
@env.task(cache="disable")
async def always_fresh_task_2(data: str) -> str:
    return get_current_timestamp() + transform_data(data)
# {{/docs-fragment disable-shorthand}}


# {{docs-fragment ignored}}
@env.task(cache=flyte.Cache(behavior="auto", ignored_inputs=("debug_flag")))
async def selective_caching(data: str, debug_flag: bool) -> str:
    if debug_flag:
        print(f"Debug: transforming {data}")
    return transform_data(data)
# {{/docs-fragment ignored}}


# {{docs-fragment serialize}}
@env.task(cache=flyte.Cache(behavior="auto", serialize=True))
async def expensive_model_training(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment serialize}}


# {{docs-fragment salt}}
@env.task(cache=flyte.Cache(behavior="auto", salt="experiment_2024_q4"))
async def experimental_analysis(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment salt}}

# {{docs-fragment policy}}
from flyte._cache import FunctionBodyPolicy


@env.task(cache=flyte.Cache(
    behavior="auto",
    policies=[FunctionBodyPolicy()]  # This is the default. Does not actually need to be specified.
))
async def code_sensitive_task(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment policy}}


# {{docs-fragment custom-policy}}
from flyte._cache import CachePolicy, VersionParameters


class DatasetVersionPolicy(CachePolicy):
    def get_version(self, salt: str, params: VersionParameters) -> str:
        # Generate version based on custom logic
        dataset_version = get_dataset_version()
        return f"{salt}_{dataset_version}"


@env.task(cache=flyte.Cache(behavior="auto", policies=[DatasetVersionPolicy()]))
async def dataset_dependent_task(data: str) -> str:
    # Cache invalidated when dataset version changes
    return transform_data(data)
# {{/docs-fragment custom-policy}}


# {{docs-fragment env-level}}
env = flyte.TaskEnvironment(
    name="cached_environment",
    cache=flyte.Cache(behavior="auto")  # Default for all tasks
)


@env.task  # Inherits auto caching from environment
async def inherits_caching(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment env-level}}


# {{docs-fragment decorator-level}}
@env.task(cache=flyte.Cache(behavior="disable"))  # Override environment default
async def decorator_caching(data: str) -> str:
    return transform_data(data)
# {{/docs-fragment decorator-level}}


# {{docs-fragment override-level}}
@env.task
async def override_caching_on_call(data: str) -> str:
    return inherits_caching(data).override(cache=flyte.Cache(behavior="disable"))
# {{/docs-fragment override-level}}


@env.task
async def main():
    data = "abcdefghijklmnopqrstuvwxyz"
    await auto_versioned_task(data)
    await auto_versioned_task_2(data)
    await manually_versioned_task(data)
    await always_fresh_task(data)
    await always_fresh_task_2(data)
    await selective_caching(data, debug_flag=True)
    await expensive_model_training(data)
    await experimental_analysis(data)
    await code_sensitive_task(data)
    await dataset_dependent_task(data)
    await inherits_caching(data)
    await decorator_caching(data)
    await override_caching_on_call(data)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()