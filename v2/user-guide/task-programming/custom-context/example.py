import flyte

env = flyte.TaskEnvironment("hello_world")

# {{docs-fragment run-context}}
@env.task
async def leaf_task():
    # Reads run-level context
    print("leaf sees:", flyte.ctx().custom_context)
    return flyte.ctx().custom_context.get("trace_id")

@env.task
async def root():
    return await leaf_task()
# {{/docs-fragment run-context}}

# {{docs-fragment override-context}}
@env.task
async def downstream():
    print("downstream sees:", flyte.ctx().custom_context)
    return flyte.ctx().custom_context.get("trace_id")

@env.task
async def parent():
    print("parent initial:", flyte.ctx().custom_context)

    # Override the trace_id for the nested call(s)
    with flyte.custom_context(trace_id="child-override"):
        val = await downstream()     # downstream sees trace_id="child-override"

    # After the context block, run-level values are back
    print("parent after:", flyte.ctx().custom_context)
    return val
# {{/docs-fragment override-context}}

# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    # Base context for the entire run
    flyte.with_runcontext(custom_context={"trace_id": "root-abc", "experiment": "v1"}).run(root)
# {{/docs-fragment main}}