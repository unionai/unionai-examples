from flytekit import task, workflow


@task
def clear_staging_table() -> None:
    # Side effect only: truncate the staging table.
    print("cleared staging table")


@task
def load_into_staging() -> None:
    # Side effect only: load fresh rows into staging.
    print("loaded staging table")


@task
def publish_to_prod() -> None:
    # Side effect only: swap staging into the production table.
    print("published to prod")


@workflow
def main() -> None:
    clear = clear_staging_table()
    load = load_into_staging()
    publish = publish_to_prod()

    # These tasks pass no data between them, so use the >> operator to force
    # ordering: clear must finish before load, which must finish before publish.
    clear >> load >> publish
