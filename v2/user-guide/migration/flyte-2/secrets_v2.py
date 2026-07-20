# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import os

import flyte

# Secrets are declared on the TaskEnvironment and injected as environment
# variables (instead of read through current_context().secrets).
env = flyte.TaskEnvironment(
    name="secrets",
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
)


@env.task
def call_api() -> str:
    token = os.getenv("OPENAI_API_KEY", "")
    return f"token has {len(token)} chars"


@env.task
def main() -> str:
    return call_api()
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
