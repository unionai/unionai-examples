import pathlib
import flyte
from pyproject_package.tasks.tasks import pipeline


def main():
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent.parent)
    run = flyte.run(pipeline, api_url="https://api.example.com/data")
    print(f"Run URL: {run.url}")
    run.wait()


def main_full():
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent.parent)
    example_url = "https://jsonplaceholder.typicode.com/posts"
    print("Starting data pipeline...")
    print(f"Target API: {example_url}")
    run = flyte.run(pipeline, api_url=example_url)
    print(f"\nRun Name: {run.name}")
    print(f"Run URL: {run.url}")
    run.wait()


if __name__ == "__main__":
    main()
