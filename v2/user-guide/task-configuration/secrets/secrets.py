# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "n=500"
# ///

import flyte
import os

# {{docs-fragment literal}}
env_1 = flyte.TaskEnvironment(
    name="env_1",
    secrets=[
        flyte.Secret(key="my_secret", as_env_var="MY_SECRET_ENV_VAR"),
    ]
)


@env_1.task
def task_1():
    my_secret_value = os.getenv("MY_SECRET_ENV_VAR")
    print(f"My secret value is: {my_secret_value}")
# {{/docs-fragment literal}}


# {{docs-fragment file}}
env_2 = flyte.TaskEnvironment(
    name="env_2",
    secrets=[
        flyte.Secret(key="my_secret", mount="/etc/flyte/secrets"),
    ]
)


@env_2.task
def task_2():
    with open("/etc/flyte/secrets/my_secret", "r") as f:
        my_secret_file_content = f.read()
    print(f"My secret file content is: {my_secret_file_content}")
# {{/docs-fragment file}}


# {{docs-fragment main}}
env = flyte.TaskEnvironment(
    name="env"
)


@env.task
def main():
    task_1()
    task_2()
# {{/docs-fragment main}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}
