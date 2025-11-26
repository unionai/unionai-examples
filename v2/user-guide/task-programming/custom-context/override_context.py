# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "parent"
# params = ""
# ///

import flyte

env = flyte.TaskEnvironment("custom-context-example")

# {{docs-fragment override-context}}
@env.task
async def downstream() -> str:
    print("downstream sees:", flyte.ctx().custom_context)
    return flyte.ctx().custom_context.get("trace_id")

@env.task
async def parent() -> str:
    print("parent initial:", flyte.ctx().custom_context)

    # Override the trace_id for the nested call(s)
    with flyte.custom_context(trace_id="child-override"):
        val = await downstream()     # downstream sees trace_id="child-override"

    # After the context block, run-level values are back
    print("parent after:", flyte.ctx().custom_context)
    return val
# {{/docs-fragment override-context}}

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(custom_context={"trace_id": "root-abc"}).run(parent)