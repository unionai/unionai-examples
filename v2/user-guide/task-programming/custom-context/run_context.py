# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "root"
# params = ""
# ///

# {{docs-fragment run-context}}
import flyte

env = flyte.TaskEnvironment("custom-context-example")

@env.task
async def leaf_task() -> str:
    # Reads run-level context
    print("leaf sees:", flyte.ctx().custom_context)
    return flyte.ctx().custom_context.get("trace_id")

@env.task
async def root() -> str:
    return await leaf_task()

if __name__ == "__main__":
    flyte.init_from_config()
    # Base context for the entire run
    flyte.with_runcontext(custom_context={"trace_id": "root-abc", "experiment": "v1"}).run(root)
# {{/docs-fragment run-context}}
