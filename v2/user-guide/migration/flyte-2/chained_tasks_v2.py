# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="dataset_lifecycle")


@env.task
def download_dataset() -> str:
    return "s3://datasets/train.parquet"


@env.task
def validate_dataset(uri: str) -> str:
    # e.g. check schema and row counts
    return f"validated {uri}"


@env.task
def register_dataset(uri: str) -> str:
    return f"registered {uri}"


# Sequential calls are naturally ordered: each line runs after the previous one
# returns. The Flyte 1 `>>` ordering operator is gone.
@env.task
def main() -> str:
    uri = download_dataset()
    validated = validate_dataset(uri)
    return register_dataset(validated)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
