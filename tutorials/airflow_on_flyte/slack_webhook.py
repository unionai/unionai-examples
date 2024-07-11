from flytekit import task, workflow

"""
To run it locally, make sure to export the following environment variables:
export AIRFLOW_CONN_UNION_SLACK='slackwebhook://<SLACK_WEBHOOK_URL>'

Also, Make sure to install apache-airflow-providers-slack locally
    pip install apache-airflow-providers-slack
"""


@task(
)
def train():
    print("Training a model...")
    print("Done.")


@workflow
def wf():
    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

    t1 = train()
    slack = SlackWebhookOperator(
        task_id="slack-webhook",
        slack_webhook_conn_id="union_slack",
        message="Training is done!",
        channel="public-demo"
    )
    t1 >> slack
