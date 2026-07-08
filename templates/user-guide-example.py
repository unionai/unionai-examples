# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    # Add every third-party package this script imports, pinned as needed.
# ]
# main = "main"           # name of the entrypoint task the test harness runs
# params = ""             # space-separated key=value defaults, e.g. "x=5 name=foo"
# ///

# Template: a single focused user-guide example.
#
# Copy this file to v2/user-guide/<area>/<feature>.py and adapt it. Keep it small,
# runnable, and heavily commented — it demonstrates ONE feature or pattern and is
# embedded into the matching User Guide page. See CONTRIBUTING.md for the rules and
# the examples-vs-product-docs boundary.
#
# Wrap any region a docs page will embed in matching fragment markers:
#     # {{docs-fragment <name>}} ... # {{/docs-fragment <name>}}
# Fragment markers are plain comments and do not affect execution.

# {{docs-fragment import-and-env}}
import flyte

# Group task configuration in a TaskEnvironment. Give it a descriptive name.
env = flyte.TaskEnvironment(
    name="example_env",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
)
# {{/docs-fragment import-and-env}}


# {{docs-fragment task}}
# Tasks are decorated with @env.task. Type annotations on the signature are required.
@env.task
def main(x: int = 5) -> int:
    """Explain, in one line, what this task demonstrates."""
    return x * 2
# {{/docs-fragment task}}


# A runnable example must have BOTH a __main__ guard AND a flyte.init... call —
# that is how the test harness discovers it as a test.
# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, x=10)
    print(run.name)
    print(run.url)
    run.wait()
# {{/docs-fragment run}}
