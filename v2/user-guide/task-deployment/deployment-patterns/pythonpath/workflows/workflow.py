import pathlib
import flyte
from src.my_module import say_hello

@flyte.task
def greet(name: str) -> str:
    return say_hello(name)

if __name__ == "__main__":
    current_dir = pathlib.Path(__file__).parent

    # Set root_dir to project root for proper import resolution
    flyte.init_from_config(
        root_dir=current_dir.parent  # Points to pythonpath/
    )

    run = flyte.run(greet, name="World")
    print(run.url)

# Alternative implementation with task environment
from src.my_module import env

workflow_env = flyte.TaskEnvironment(
    name="workflow_env",
    depends_on=[env],
)

@workflow_env.task
async def greet_async(name: str) -> str:
    return await say_hello(name)
