# {{docs-fragment pythonpath-module}}
def say_hello(name: str) -> str:
    """Business logic with no Flyte dependencies"""
    return f"Hello, {name}!"
# {{/docs-fragment pythonpath-module}}

# Flyte task environment for this module
import flyte

env = flyte.TaskEnvironment(
    name="my_module",
)

@env.task
async def say_hello_task(name: str) -> str:
    """A simple task that greets the user."""
    return say_hello(name)
