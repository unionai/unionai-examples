from flytekit import task, workflow, LaunchPlan, CronSchedule


@task
def retrain(kickoff_time: str) -> str:
    return f"retrained model at {kickoff_time}"


@workflow
def main(kickoff_time: str) -> str:
    return retrain(kickoff_time=kickoff_time)


# A LaunchPlan attaches a schedule (and default inputs) to a workflow.
nightly_retrain = LaunchPlan.get_or_create(
    workflow=main,
    name="nightly_retrain",
    schedule=CronSchedule(
        schedule="0 2 * * *",  # 2 AM daily
        kickoff_time_input_arg="kickoff_time",
    ),
)
