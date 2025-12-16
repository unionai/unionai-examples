import pathlib
import flyte
from pyproject_package.tasks.tasks import pipeline


def main():
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent.parent)
    run = flyte.run(pipeline, api_url="https://api.example.com/data")
    print(f"Run URL: {run.url}")
    run.wait()

if __name__ == "__main__":
    main()
