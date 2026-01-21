import asyncio
from datetime import timedelta


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

    results = await asyncio.gather(*agent_tasks)
    return sweep_id
