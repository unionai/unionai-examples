import flyte

from flyteplugins.wandb import get_wandb_run, wandb_config, wandb_init

env = flyte.TaskEnvironment(
    name="wandb-example",
    image=flyte.Image.from_debian_base(name="wandb-example").with_pip_packages(
        "flyteplugins-wandb"
    ),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
@env.task
async def train_model() -> str:
    wandb_run = get_wandb_run()

    # Your training code here
    for epoch in range(10):
        loss = 1.0 / (epoch + 1)
        wandb_run.log({"epoch": epoch, "loss": loss})

    return "Training complete"


if __name__ == "__main__":
    flyte.init_from_config()

    r = flyte.with_runcontext(
        custom_context=wandb_config(
            project="my-project",
            entity="my-team",
        ),
    ).run(train_model)

    print(f"run url: {r.url}")
