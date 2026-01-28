import flyte
from flyte.io import DataFrame
from flyteplugins.connectors.snowflake import Snowflake, SnowflakeConfig

config = SnowflakeConfig(
    account="myorg-myaccount",
    user="flyte_user",
    database="ANALYTICS",
    schema="PUBLIC",
    warehouse="COMPUTE_WH",
    connection_kwargs={
        "role": "ETL_ROLE",
    },
)

# Step 1: Create the staging table if it doesn't exist
create_staging = Snowflake(
    name="create_staging",
    query_template="""
        CREATE TABLE IF NOT EXISTS staging.daily_events (
            event_id STRING,
            event_date DATE,
            user_id STRING,
            event_type STRING,
            payload VARIANT
        )
    """,
    plugin_config=config,
    snowflake_private_key="snowflake",
    snowflake_private_key_passphrase="snowflake_passphrase",
)

# Step 2: Insert a record into the staging table
insert_events = Snowflake(
    name="insert_event",
    query_template="""
        INSERT INTO staging.daily_events (event_id, event_date, user_id, event_type)
        VALUES (%(event_id)s, %(event_date)s, %(user_id)s, %(event_type)s)
    """,
    plugin_config=config,
    inputs={
        "event_id": list[str],
        "event_date": list[str],
        "user_id": list[str],
        "event_type": list[str],
    },
    snowflake_private_key="snowflake",
    snowflake_private_key_passphrase="snowflake_passphrase",
    batch=True,
)

# Step 3: Query aggregated results for a given date
daily_summary = Snowflake(
    name="daily_summary",
    query_template="""
        SELECT
            event_type,
            COUNT(*) AS event_count,
            COUNT(DISTINCT user_id) AS unique_users
        FROM staging.daily_events
        WHERE event_date = %(report_date)s
        GROUP BY event_type
        ORDER BY event_count DESC
    """,
    plugin_config=config,
    inputs={"report_date": str},
    output_dataframe_type=DataFrame,
    snowflake_private_key="snowflake",
    snowflake_private_key_passphrase="snowflake_passphrase",
)

# Create environments for all Snowflake tasks
snowflake_env = flyte.TaskEnvironment.from_task(
    "snowflake_env", create_staging, insert_events, daily_summary
)

# Main pipeline environment, depends on the Snowflake task environments
env = flyte.TaskEnvironment(
    name="analytics_env",
    resources=flyte.Resources(memory="512Mi"),
    image=flyte.Image.from_debian_base(name="analytics").with_pip_packages(
        "flyteplugins-connectors[snowflake]", "pandas", pre=True
    ),
    secrets=[
        flyte.Secret(key="snowflake", as_env_var="SNOWFLAKE_PRIVATE_KEY"),
        flyte.Secret(
            key="snowflake_passphrase", as_env_var="SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"
        ),
    ],
    depends_on=[snowflake_env],
)


# Step 4: Process the results in Python
@env.task
async def generate_report(summary: DataFrame) -> dict:
    import pandas as pd

    df = await summary.open(pd.DataFrame).all()
    total_events = df["event_count"].sum()
    top_event = df.iloc[0]["event_type"]
    return {
        "total_events": int(total_events),
        "top_event_type": top_event,
        "event_types_count": len(df),
    }


# Compose the pipeline
@env.task
async def run_daily_pipeline(
    event_ids: list[str],
    event_dates: list[str],
    user_ids: list[str],
    event_types: list[str],
) -> dict:
    await create_staging()
    await insert_events(
        event_id=event_ids,
        event_date=event_dates,
        user_id=user_ids,
        event_type=event_types,
    )
    summary = await daily_summary(report_date=event_dates[0])
    return await generate_report(summary=summary)


if __name__ == "__main__":
    flyte.init_from_config()

    # Run locally
    run = flyte.with_runcontext(mode="local").run(
        run_daily_pipeline,
        event_ids=["event-1", "event-2"],
        event_dates=["2023-01-01", "2023-01-02"],
        user_ids=["user-1", "user-2"],
        event_types=["click", "view"],
    )

    # Or run remotely
    run = flyte.with_runcontext(mode="remote").run(
        run_daily_pipeline,
        event_ids=["event-1", "event-2"],
        event_dates=["2023-01-01", "2023-01-02"],
        user_ids=["user-1", "user-2"],
        event_types=["click", "view"],
    )

    print(run.url)
