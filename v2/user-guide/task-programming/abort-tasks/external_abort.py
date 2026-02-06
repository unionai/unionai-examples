# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "n = 10, sleep_for = 30.0"
# ///

import asyncio

import flyte
import flyte.errors

env = flyte.TaskEnvironment("external_abort")


@env.task
async def long_sleeper(sleep_for: float):
    await asyncio.sleep(sleep_for)


@env.task
async def main(n: int, sleep_for: float) -> str:
    coros = [long_sleeper(sleep_for) for _ in range(n)]
    results = await asyncio.gather(*coros, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, flyte.errors.ActionAbortedError):
            print(f"Action [{i}] was externally aborted")
    return "Hello World!"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, 10, 30.0)
    print(run.url)
    run.wait()
