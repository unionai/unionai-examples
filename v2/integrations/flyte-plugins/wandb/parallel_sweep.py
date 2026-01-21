import asyncio
from datetime import timedelta

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
    name="wandb-parallel-sweep-example",
    image=flyte.Image.from_debian_base(
        name="wandb-parallel-sweep-example"
    ).with_pip_packages("flyteplugins-wandb"),
    secrets=[flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")],
)


@wandb_init
def objective():
    run = wandb.run
    config = run.config

    for epoch in range(config.epochs):
        loss = 1.0 / (config.learning_rate * config.batch_size) + epoch * 0.1
        run.log({"epoch": epoch, "loss": loss})


@wandb_sweep
@env.task
async def sweep_agent(agent_id: int, sweep_id: str, count: int = 5) -> int:
    """Single agent that runs a subset of trials."""
    wandb.agent(sweep_id, function=objective, count=count)
    return agent_id


@wandb_sweep
@env.task
async def run_parallel_sweep(total_trials: int = 20, trials_per_agent: int = 5) -> str:
    """Orchestrate multiple agents running in parallel."""
    sweep_id = get_wandb_sweep_id()

    num_agents = (total_trials + trials_per_agent - 1) // trials_per_agent

    # Launch agents in parallel, each with its own resources
    agent_tasks = [
        sweep_agent.override(
            resources=flyte.Resources(cpu="2", memory="4Gi"),
            retries=3,
            timeout=timedelta(minutes=30),
        )(agent_id=i, sweep_id=sweep_id, count=trials_per_agent)
        for i in range(num_agents)
    ]

    await asyncio.gather(*agent_tasks)
    return sweep_id


if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.with_runcontext(
        custom_context={
            **wandb_config(project="my-project", entity="my-team"),
            **wandb_sweep_config(
                method="random",
                metric={"name": "loss", "goal": "minimize"},
                parameters={
                    "learning_rate": {"min": 0.0001, "max": 0.1},
                    "batch_size": {"values": [16, 32, 64]},
                    "epochs": {"values": [5, 10, 20]},
                },
            ),
        },
    ).run(
        run_parallel_sweep,
        total_trials=20,
        trials_per_agent=5,
    )

    print(f"sweep url: {run.url}")
