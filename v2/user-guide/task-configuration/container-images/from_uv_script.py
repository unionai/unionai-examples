# {{docs-fragment from-uv-script}}
# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "flyte",
#    "numpy",
#    "pandas",
#    "scikit-learn"
# ]
# ///

...

env = flyte.TaskEnvironment(
    name="my_env",
    image=flyte.Image.from_uv_script(
            __file__,
            name="my_image",
            registry="ghcr.io/my_gh_org" # Only needed for local builds
        )
)

# Supporting task definitions
...

# Main task definition
@env.task
def main(x_list: list[int] = list(range(10))) -> float:
    ...

# Init and run
if __name__ == "__main__":
    # Init for remote run on backend
    flyte.init_from_config("config.yaml")

    # Init for local run
    # flyte.init()

    run = flyte.run(main, x_list=list(range(10)))
    print(run.name)
    print(run.url)
    run.wait(run)
# {{/docs-fragment from-uv-script}}