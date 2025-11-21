# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# ///

# {{docs-fragment all}}
import flyte
from flyte import remote

env_1 = flyte.TaskEnvironment(name="env_1")
env_2 = flyte.TaskEnvironment(name="env_2")
env_1.add_dependency(env_2)


@env_2.task
async def remote_task(x: str) -> str:
    return "Remote task processed: " + x


@env_1.task
async def main() -> str:
    remote_task_ref = remote.Task.get("env_2.remote_task", auto_version="latest")
    r = await remote_task_ref(x="Hello")
    return "main called remote and recieved: " + r


if __name__ == "__main__":
    flyte.init_from_config()
    d = flyte.deploy(env_1)
    print(d[0].summary_repr())
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}