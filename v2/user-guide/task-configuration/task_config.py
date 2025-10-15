# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = ""
# ///

import flyte

another_env = flyte.TaskEnvironment(name="another_env")

# {{docs-fragment simple}}
env = flyte.TaskEnvironment(name="my_env")

@env.task
async def my_task(name:str) -> str:
    return f"Hello {name}!"
# {{docs-fragment simple}}

# {{docs-fragment config-levels}}
# Level 1: TaskEnvironment - Base configuration
env_2 = flyte.TaskEnvironment(
    name="data_processing_env",
    image=flyte.Image.from_debian_base(),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    env_vars={"MY_VAR": "value"},
    # secrets=flyte.Secret(key="openapi_key", as_env_var="MY_API_KEY"),
    cache="disable",
    # pod_template=my_pod_template,
    # reusable=flyte.ReusePolicy(replicas=2, idle_ttl=300),
    depends_on=[another_env],
    description="Data processing task environment",
    # plugin_config=my_plugin_config
)

# Level 2: Decorator - Override some environment settings
@env_2.task(
    short_name="process",
    # secrets=flyte.Secret(key="openapi_key", as_env_var="MY_API_KEY_2"),
    cache="auto",
    # pod_template=my_pod_template,
    report=True,
    max_inline_io_bytes=100 * 1024,
    retries=3,
    timeout=60,
    docs="This task processes data and generates a report."
)
async def process_data(data_path: str) -> str:
    return f"Processed {data_path}"

@env_2.task
async def invoke_process_data() -> str:
    result = await process_data.override(
        resources=flyte.Resources(cpu=4, memory="2Gi"),
        env_vars={"MY_VAR": "new_value"},
        # secrets=flyte.Secret(key="openapi_key", as_env_var="MY_API_KEY_3"),
        cache="auto",
        max_inline_io_bytes=100 * 1024,
        retries=3,
        timeout=60
    )("input.csv")
    return result
# {{docs-fragment config-levels}}


@env_2.task
async def main():
    greeting = await my_task("World")
    print(greeting)
    processed = await process_data("input.csv")
    print(processed)
    processed = await invoke_process_data()
    print(processed)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.name)
    print(run.url)
    run.wait()