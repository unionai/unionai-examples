# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
# ]
# ///

# {{docs-fragment all}}
# https://github.com/unionai/unionai-examples/blob/main/v2/user-guide/flyte-2/remote.py

import flyte
from flyte import remote

env_1 = flyte.TaskEnvironment(name="env_1")
env_2 = flyte.TaskEnvironment(name="env_2")
env_1.add_dependency(env_2)


@env_2.task
async def remote_task(x: str) -> str:
    return "Remote task processed: " + x


@env_1.task
async def main(remote_task: remote.Task) -> str:
    r = await remote_task(x="Hello")
    return "main called remote and recieved: " + r


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(env_1)
    print(d[0].summary_repr())
    remote_task = remote.Task.get("env_2.remote_task", auto_version="latest")
    r = flyte.run(main, remote_task=remote_task)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}