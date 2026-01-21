import flyte
import wandb
from flyteplugins.wandb import WandbSweep

env = flyte.TaskEnvironment(
    name="wandb-manual-sweep-example",
    image=flyte.Image.from_debian_base(
        name="wandb-manual-sweep-example"
    ).with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


def objective():
    with wandb.init(project="my-project", entity="my-team") as run:
        config = run.config

        for epoch in range(config.epochs):
            loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
            run.log({"epoch": epoch, "loss": loss})


@env.task(
    links=(
        WandbSweep(
            project="my-project",
            entity="my-team",
        ),
    )
)
async def manual_sweep() -> str:
    # Manually create the sweep
    sweep_config = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1},
            "batch_size": {"values": [16, 32, 64]},
            "epochs": {"value": 10},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="my-project", entity="my-team")

    # Run the sweep
    wandb.agent(sweep_id, function=objective, count=10, project="my-project")

    return sweep_id


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext().run(manual_sweep)

    print(f"run url: {run.url}")
