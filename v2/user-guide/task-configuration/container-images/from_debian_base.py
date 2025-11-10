import flyte
import numpy as np

# Define the task environment
env = flyte.TaskEnvironment(
    name="my_env",
    image = (
        flyte.Image.from_debian_base(
            name="my-image",
            python_version=(3, 13)
            # registry="registry.example.com/my-org" # Only needed for local builds
        )
        .with_apt_packages("libopenblas-dev")
        .with_pip_packages("numpy")
        .with_env_vars({"OMP_NUM_THREADS": "4"})
    )
)


@env.task
def main(x_list: list[int]) -> float:
    arr = np.array(x_list)
    return float(np.mean(arr))


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, x_list=list(range(10)))
    print(r.name)
    print(r.url)
    r.wait()
