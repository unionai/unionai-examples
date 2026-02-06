# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "n=500"
# ///

# {{docs-fragment hourly}}
import flyte
from datetime import datetime, timezone

env = flyte.TaskEnvironment(name="trigger_env")

@env.task(triggers=flyte.Trigger.hourly())  # Every hour
def hourly_task(trigger_time: datetime, x: int = 1) -> str:
    return f"Hourly example executed at {trigger_time.isoformat()} with x={x}"
# {{/docs-fragment hourly}}

name="dummy_trigger"
automation=flyte.Trigger.daily()  # Daily trigger
# {{docs-fragment dummy-trigger}}
flyte.Trigger(
    name,
    automation,
    description="",
    auto_activate=True,
    inputs=None,
    env_vars=None,
    interruptible=None,
    overwrite_cache=False,
    queue=None,
    labels=None,
    annotations=None
)
# {{/docs-fragment dummy-trigger}}

# {{docs-fragment comprehensive-trigger}}
comprehensive_trigger = flyte.Trigger(
    name="monthly_financial_report",
    automation=flyte.Cron("0 6 1 * *", timezone="America/New_York"),
    description="Monthly financial report generation for executive team",
    auto_activate=True,
    inputs={
        "report_date": flyte.TriggerTime,
        "report_type": "executive_summary",
        "include_forecasts": True
    },
    env_vars={
        "REPORT_OUTPUT_FORMAT": "PDF",
        "EMAIL_NOTIFICATIONS": "true"
    },
    interruptible=False,  # Critical report, use dedicated resources
    overwrite_cache=True,  # Always fresh data
    queue="financial-reports",
    labels={
        "team": "finance",
        "criticality": "high",
        "automation": "scheduled"
    },
    annotations={
        "compliance.company.com/sox-required": "true",
        "backup.company.com/retain-days": "2555"  # 7 years
    }
)
# {{/docs-fragment comprehensive-trigger}}

interval_minutes = 15
# {{docs-fragment dummy-fixed-rate}}
flyte.FixedRate(
    interval_minutes,
    start_time=None
)
# {{/docs-fragment dummy-fixed-rate}}

# {{docs-fragment fixed-rate-examples}}
# Every 90 minutes, starting when deployed
every_90_min = flyte.Trigger(
    "data_processing",
    flyte.FixedRate(interval_minutes=90)
)

# Every 6 hours (360 minutes), starting at a specific time
specific_start = flyte.Trigger(
    "batch_job",
    flyte.FixedRate(
        interval_minutes=360,  # 6 hours
        start_time=datetime(2025, 12, 1, 9, 0, 0)  # Start Dec 1st at 9 AM
    )
)
# {{/docs-fragment fixed-rate-examples}}

cron_expression = "0 0 * * *"
# {{docs-fragment dummy-cron}}
flyte.Cron(
    cron_expression,
    timezone=None
)
# {{/docs-fragment dummy-cron}}

# {{docs-fragment cron-examples}}
# Every day at 6 AM UTC
daily_trigger = flyte.Trigger(
    "daily_report",
    flyte.Cron("0 6 * * *")
)

# Every weekday at 9:30 AM Eastern Time
weekday_trigger = flyte.Trigger(
    "business_hours_task",
    flyte.Cron("30 9 * * 1-5", timezone="America/New_York")
)
# {{/docs-fragment cron-examples}}

# {{docs-fragment inputs-basic-usage}}
trigger_with_inputs = flyte.Trigger(
    "data_processing",
    flyte.Cron("0 6 * * *"),  # Daily at 6 AM
    inputs={
        "batch_size": 1000,
        "environment": "production",
        "debug_mode": False
    }
)

@env.task(triggers=trigger_with_inputs)
def process_data(batch_size: int, environment: str, debug_mode: bool = True) -> str:
    return f"Processing {batch_size} items in {environment} mode"
# {{/docs-fragment inputs-basic-usage}}

# {{docs-fragment inputs-trigger-time}}
timestamp_trigger = flyte.Trigger(
    "daily_report",
    flyte.Cron("0 0 * * *"),  # Daily at midnight
    inputs={
        "report_date": flyte.TriggerTime,  # Receives trigger execution time
        "report_type": "daily_summary"
    }
)

@env.task(triggers=timestamp_trigger)
def generate_report(report_date: datetime, report_type: str) -> str:
    return f"Generated {report_type} for {report_date.strftime('%Y-%m-%d')}"
# {{/docs-fragment inputs-trigger-time}}

# {{docs-fragment inputs-complex}}
complex_trigger = flyte.Trigger(
    "ml_training",
    flyte.Cron("0 2 * * 1"),  # Weekly on Monday at 2 AM
    inputs={
        "model_config": {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100
        },
        "feature_columns": ["age", "income", "location"],
        "validation_split": 0.2,
        "training_date": flyte.TriggerTime
    }
)

@env.task(triggers=complex_trigger)
def train_model(
    model_config: dict,
    feature_columns: list[str],
    validation_split: float,
    training_date: datetime
) -> str:
    return f"Training model with {len(feature_columns)} features on {training_date}"
# {{/docs-fragment inputs-complex}}

# {{docs-fragment predefined-available}}
minutely_trigger = flyte.Trigger.minutely()    # Every minute
hourly_trigger = flyte.Trigger.hourly()        # Every hour
daily_trigger = flyte.Trigger.daily()          # Every day at midnight
weekly_trigger = flyte.Trigger.weekly()        # Every week (Sundays at midnight)
monthly_trigger = flyte.Trigger.monthly()      # Every month (1st day at midnight)
# {{/docs-fragment predefined-available}}

# {{docs-fragment predefined-parameters}}
flyte.Trigger.daily(
    trigger_time_input_key="trigger_time",
    name="daily",
    description="A trigger that runs daily at midnight",
    auto_activate=True,
    inputs=None,
    env_vars=None,
    interruptible=None,
    overwrite_cache=False,
    queue=None,
    labels=None,
    annotations=None
)
# {{/docs-fragment predefined-parameters}}

# {{docs-fragment trigger-time-input-key}}
@env.task(triggers=flyte.Trigger.daily(trigger_time_input_key="scheduled_at"))
def task_with_custom_trigger_time_input(scheduled_at: datetime) -> str:
    return f"Executed at {scheduled_at}"
# {{/docs-fragment trigger-time-input-key}}

# {{docs-fragment multiple-triggers}}
@env.task(triggers=[
    flyte.Trigger.hourly(),  # Predefined trigger
    flyte.Trigger.daily(),   # Another predefined trigger
    flyte.Trigger("custom", flyte.Cron("0 */6 * * *"))  # Custom trigger every 6 hours
])
def multi_trigger_task(trigger_time: datetime = flyte.TriggerTime) -> str:
    # Different logic based on execution timing
    if trigger_time.hour == 0:  # Daily run at midnight
        return f"Daily comprehensive processing at {trigger_time}"
    else:  # Hourly or custom runs
        return f"Regular processing at {trigger_time.strftime('%H:%M')}"
# {{/docs-fragment multiple-triggers}}

# {{docs-fragment auto-activate-false}}
env = flyte.TaskEnvironment(name="my_task_env")

custom_cron_trigger = flyte.Trigger(
    "custom_cron",
    flyte.Cron("0 0 * * *"),
    auto_activate=False # Dont create runs yet
)

@env.task(triggers=custom_cron_trigger)
def custom_task() -> str:
    return "Hello, world!"
# {{/docs-fragment auto-activate-false}}

# {{docs-fragment no-start-time-auto-activate-true}}
my_trigger = flyte.Trigger("my_trigger", flyte.FixedRate(60))
# {{/docs-fragment no-start-time-auto-activate-true}}

# {{docs-fragment no-start-time-auto-activate-false}}
my_trigger = flyte.Trigger("my_trigger", flyte.FixedRate(60), auto_activate=False)
# {{/docs-fragment no-start-time-auto-activate-false}}

# {{docs-fragment fixed-rate-with-start-time-while-active}}
my_trigger = flyte.Trigger(
    "my_trigger",
    # Runs every 60 minutes starting from October 26th, 2025, 10:00am
    flyte.FixedRate(60, start_time=datetime(2025, 10, 26, 10, 0, 0)),
)
# {{/docs-fragment fixed-rate-with-start-time-while-active}}

# {{docs-fragment fixed-rate-with-start-time-while-inactive}}
custom_rate_trigger = flyte.Trigger(
    "custom_rate",
    # Runs every 60 minutes starting from October 26th, 2025, 10:00am
    flyte.FixedRate(60, start_time=datetime(2025, 10, 26, 10, 0, 0)),
    auto_activate=False
)
# {{/docs-fragment fixed-rate-with-start-time-while-inactive}}

# {{docs-fragment timezone}}
sf_trigger = flyte.Trigger(
    "sf_tz",
    flyte.Cron(
        "0 9 * * *", timezone="America/Los_Angeles"
    ), # Every day at 9 AM PT
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)

nyc_trigger = flyte.Trigger(
    "nyc_tz",
    flyte.Cron(
        "1 12 * * *", timezone="America/New_York"
    ), # Every day at 12:01 PM ET
    inputs={"start_time": flyte.TriggerTime, "x": 1},
)
# {{/docs-fragment timezone}}

# {{docs-fragment trigger-time-utc}}
@env.task(triggers=flyte.Trigger.minutely(trigger_time_input_key="utc_trigger_time", name="timezone_trigger"))
def timezone_task(utc_trigger_time: datetime) -> str:
    local_time = utc_trigger_time.replace(tzinfo=timezone.utc).astimezone()
    return f"Task fired at {utc_trigger_time} UTC ({local_time} local)"
# {{/docs-fragment trigger-time-utc}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(env)
# {{/docs-fragment deploy}}