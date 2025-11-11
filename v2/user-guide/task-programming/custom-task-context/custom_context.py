# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
# ]
# main = "main"
# params = "x=10"
# ///

import flyte

env = flyte.TaskEnvironment("custom_context")


# {{docs-fragment downstream-task}}
@env.task
async def downstream_task(x: int) -> int:
    custom_ctx = flyte.ctx().custom_context
    if "increment" not in custom_ctx:
        raise ValueError("Expected 'increment' in custom context")
    return x + int(custom_ctx["increment"])
# {{/docs-fragment downstream-task}}


# {{docs-fragment main-task}}
@env.task
async def main(x: int) -> int:
    vals = []
    for i in range(3):
        with flyte.custom_context(increment=str(i)):
            vals.append(await downstream_task(x))
    return sum(vals)
# {{/docs-fragment main-task}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, x=10)
    print(r.name)
    print(r.url)
    r.wait()