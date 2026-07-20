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

env = flyte.TaskEnvironment(name="staging_publish")


@env.task
def clear_staging_table() -> None:
    print("cleared staging table")


@env.task
def load_into_staging() -> None:
    print("loaded staging table")


@env.task
def publish_to_prod() -> None:
    print("published to prod")


# Sequential (synchronous) calls run in the order they're written, even when no
# data flows between them. The Flyte 1 `>>` ordering operator is gone.
@env.task
def main() -> None:
    clear_staging_table()
    load_into_staging()
    publish_to_prod()
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
