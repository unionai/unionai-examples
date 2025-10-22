# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment task-env}}
import flyte

# Define a TaskEnvironment for ML training tasks
env = flyte.TaskEnvironment(
    name="ml-training",
    resources=flyte.Resources(
        cpu=("2", "8"),        # Request 2 cores, allow up to 8 cores for scaling
        memory=("8Gi", "32Gi"), # Request 8 GiB, allow up to 32 GiB for large datasets
        gpu="A100:2",          # 2 NVIDIA A100 GPUs for training
        disk="50Gi",           # 50 GiB ephemeral storage for checkpoints
        shm="8Gi"              # 8 GiB shared memory for efficient data loading
    )
)

# Use the environment for tasks
@env.task
async def train_model(dataset_path: str) -> str:
    # This task will run with flexible resource allocation
    return "model trained"
# {{/docs-fragment task-env}}

# {{docs-fragment override}}
# Demonstrate resource override at task invocation level
@env.task
async def heavy_training_task() -> str:
    return "heavy model trained with overridden resources"


@env.task
async def main():
    # Task using environment-level resources
    result = await train_model("data.csv")
    print(result)

    # Task with overridden resources at invocation time
    result = await heavy_training_task.override(
        resources=flyte.Resources(
            cpu="16",
            memory="64Gi",
            gpu="H100:2",
            disk="100Gi",
            shm="16Gi"
        )
    )()
    print(result)
# {{/docs-fragment override}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()