# {{docs-fragment from-debian-base}}
import flyte

# Define the task environment
env = flyte.TaskEnvironment(
    name="my_env",
    image = (
        flyte.Image.from_debian_base(
            name="my-image"
            python_version=(3, 13),
            registry="ghcr.io/my_gh_org" # Only needed for local builds
        )
        .with_apt_packages("git", "curl")
        .with_pip_packages("numpy", "pandas", "scikit-learn")
        .with_env_vars({"MY_CONFIG": "production"})
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
    flyte.init_from_config("config.yaml")
    run = flyte.run(main, x_list=list(range(10)))
    print(run.name)
    print(run.url)
    run.wait(run)
# {{/docs-fragment from-debian-base}}
