# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = ""
# ///

import asyncio

import flyte
import flyte.errors

env = flyte.TaskEnvironment(name="fail", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task
def oomer() -> int:
    """Allocate enough memory to trigger an out-of-memory error."""
    large_list = [0] * 100000000
    print(len(large_list))
    return 42


@env.task
def main() -> int:
    """Retry an out-of-memory task with more resources."""
    try:
        return oomer()
    except flyte.errors.OOMError as e:
        print(f"Failed with oom trying with more resources: {e}, of type {type(e)}, {e.code}")
        try:
            return oomer.override(resources=flyte.Resources(cpu=1, memory="1Gi"))(5)
        except flyte.errors.OOMError as e:
            print(f"Failed with OOM Again giving up: {e}, of type {type(e)}, {e.code}")
            raise e


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
