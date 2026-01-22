import flyte
import wandb
from flyteplugins.wandb import (
    get_wandb_sweep_id,
    wandb_config,
    wandb_init,
    wandb_sweep,
    wandb_sweep_config,
)

env = flyte.TaskEnvironment(
    name="wandb-example",
    image=flyte.Image.from_debian_base(name="wandb-example").with_pip_packages(
        "flyteplugins-wandb"
    ),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
def objective():
    """Objective function that W&B calls for each trial."""
    wandb_run = wandb.run
    config = wandb_run.config

    # Simulate training with hyperparameters from the sweep
    for epoch in range(config.epochs):
        loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
        wandb_run.log({"epoch": epoch, "loss": loss})


@wandb_sweep
@env.task
async def run_sweep() -> str:
    sweep_id = get_wandb_sweep_id()

    # Run 10 trials
    wandb.agent(sweep_id, function=objective, count=10)

    return sweep_id


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.with_runcontext(
        custom_context={
            **wandb_config(project="my-project", entity="my-team"),
            **wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={
                    "learning_rate": {"min": 0.0001, "max": 0.1},
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "epochs": {"values": [5, 10, 20]},
                },
            ),
        },
    ).run(run_sweep)

    print(f"run url: {r.url}")
