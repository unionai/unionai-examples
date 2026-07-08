# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    # Add every third-party package this tutorial imports, pinned as needed.
# ]
# main = "main"           # name of the entrypoint task the test harness runs
# params = ""             # space-separated key=value defaults, e.g. "companies=[AAPL] focus=earnings"
# ///

# Template: an end-to-end tutorial entrypoint.
#
# Copy the whole `templates/tutorial/` directory to v2/tutorials/<name>/ and adapt
# it. A tutorial tells one complete, runnable story. Split supporting logic into
# sibling modules if it grows; keep this file the entrypoint. Fill in README.md.
#
# Stay on the EXAMPLE side of the examples-vs-product-docs boundary (CONTRIBUTING.md):
# demonstrate and link to the product docs — don't reproduce feature reference here.

import flyte

env = flyte.TaskEnvironment(
    name="tutorial_env",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)


@env.task
def step(item: str) -> str:
    """A unit of work. Replace with the real per-item logic."""
    return item.upper()


@env.task
def main(items: list[str] = ["a", "b", "c"]) -> list[str]:
    """The entrypoint task. Orchestrates the tutorial's workflow."""
    # flyte.map runs `step` over the inputs in parallel (like Python's map).
    return list(flyte.map(step, items))


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, items=["one", "two", "three"])
    print(run.name)
    print(run.url)
    run.wait()
