import flyte
import torch
import torch.distributed
from flyteplugins.pytorch.task import Elastic
from flyteplugins.wandb import get_wandb_run, wandb_config, wandb_init

image = flyte.Image.from_debian_base(name="torch-wandb").with_pip_packages(
    "flyteplugins-wandb", "flyteplugins-pytorch"
)

env = flyte.TaskEnvironment(
    name="distributed_env",
    image=image,
    resources=flyte.Resources(gpu="A100:2"),
    plugin_config=Elastic(nproc_per_node=2, nnodes=1),
    secrets=flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
)


@wandb_init
@env.task
def train() -> float:
    torch.distributed.init_process_group("nccl")

    # Only rank 0 gets a W&B run object; others get None
    run = get_wandb_run()

    # Simulate training
    for step in range(100):
        loss = 1.0 / (step + 1)

        # Safe to call on all ranks - only rank 0 actually logs
        if run:
            run.log({"loss": loss, "step": step})

    torch.distributed.destroy_process_group()
    return loss


if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(
        custom_context=wandb_config(project="my-project", entity="my-team")
    ).run(train)
