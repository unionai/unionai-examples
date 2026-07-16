from flytekit import task, workflow


@task
def download_dataset() -> str:
    return "s3://datasets/train.parquet"


@task
def validate_dataset(uri: str) -> str:
    # e.g. check schema and row counts
    return f"validated {uri}"


@task
def register_dataset(uri: str) -> str:
    return f"registered {uri}"


@workflow
def main() -> str:
    uri = download_dataset()
    validated = validate_dataset(uri=uri)
    return register_dataset(uri=validated)
