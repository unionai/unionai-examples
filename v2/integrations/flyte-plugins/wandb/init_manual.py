import flyte
import wandb
from flyteplugins.wandb import Wandb

env = flyte.TaskEnvironment(
    name="wandb-manual-init-example",
    image=flyte.Image.from_debian_base(
        name="wandb-manual-init-example"
    ).with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@env.task(
    links=(
        Wandb(
            project="my-project",
            entity="my-team",
            run_mode="new",
            # No id parameter - link will auto-generate from run_name-action_name
        ),
    )
)
async def train_model(learning_rate: float) -> str:
    ctx = flyte.ctx()

    # Generate run ID matching the link's auto-generated ID
    run_id = f"{ctx.action.run_name}-{ctx.action.name}"

    # Manually initialize W&B
    wandb_run = wandb.init(
        project="my-project",
        entity="my-team",
        id=run_id,
        config={"learning_rate": learning_rate},
    )

    # Your training code
    for epoch in range(10):
        loss = 1.0 / (learning_rate * (epoch + 1))
        wandb_run.log({"epoch": epoch, "loss": loss})

    # Manually finish the run
    wandb_run.finish()

    return wandb_run.id


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.with_runcontext().run(
        train_model,
        learning_rate=0.01,
    )

    print(f"run url: {r.url}")
