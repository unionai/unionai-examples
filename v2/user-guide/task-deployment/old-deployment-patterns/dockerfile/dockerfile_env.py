from pathlib import Path

import flyte

env = flyte.TaskEnvironment(
    name="docker_env",
    image=flyte.Image.from_dockerfile(
        Path(__file__).parent / "Dockerfile",
        registry="ghcr.io/flyteorg",
        name="docker_env_image",
    ),
)

@env.task
def main(x: int) -> int:
    return x * 2

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, x=10)
    print(run.url)
