# container_images.py

# /// script
# dependencies = [
#    "polars",
#    "flyte>=0.2.0b27"
# ]
# ///

import polars as pl

import flyte

# Replace with your container registry URL
MY_CONTAINER_REGISTRY = "<my-container-registry>"

env = flyte.TaskEnvironment(
    name="polars_env",
    image=flyte.Image.from_uv_script(
        __file__,
        name="polars_image",
        registry=MY_CONTAINER_REGISTRY,
    ),
)


@env.task
async def create_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"name": ["Alice", "Bob", "Charlie"], "age": [25, 32, 37], "city": ["New York", "Paris", "Berlin"]}
    )


@env.task
async def print_dataframe(dataframe: pl.DataFrame):
    print(dataframe)


@env.task
async def workflow():
    df = await create_dataframe()
    await print_dataframe(df)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(workflow)
    print(run.name)
    print(run.url)
    run.wait()
