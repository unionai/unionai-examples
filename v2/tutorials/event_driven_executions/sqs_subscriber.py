# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte>=2.6.12",
#     "boto3",
#     "fastapi",
#     "uvicorn",
# ]
# ///
"""Read an SQS queue and launch a run per message, as a Union app."""

import json
import logging
import os
import threading

import boto3
import flyte
import flyte.remote as remote
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sqs")

QUEUE_URL = os.environ["QUEUE_URL"]
TASK = os.environ["FLYTE_TASK"]
RELEASE = os.environ.get("FLYTE_RELEASE", "prod")

app = FastAPI()
_alive = False


# {{docs-fragment sqs_inputs}}
def to_inputs(body: str) -> dict | None:
    """Map an S3 notification to task inputs, or None if there is nothing to act on.

    S3 delivers a JSON string wrapping a list of records. Each record describes the
    object; it is not the task's inputs, so it has to be mapped. A newly configured
    notification also sends an `s3:TestEvent`, which has no `Records` at all.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return None

    for record in payload.get("Records", []):
        if not record.get("eventName", "").startswith("ObjectCreated"):
            continue
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
        return {"object_key": f"s3://{bucket}/{key}"}
    return None
# {{/docs-fragment sqs_inputs}}


# {{docs-fragment sqs_poll}}
def poll_forever() -> None:
    global _alive
    sqs = boto3.client("sqs")
    task = remote.Task.get(TASK, version=RELEASE)
    _alive = True

    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL, MaxNumberOfMessages=10, WaitTimeSeconds=20
        )
        for message in response.get("Messages", []):
            inputs = to_inputs(message["Body"])
            if inputs is not None:
                try:
                    # The run name comes from the SQS MessageId, so a redelivery
                    # collides with the run that already exists rather than
                    # starting a second one.
                    run = flyte.with_runcontext(
                        name=f"sqs-{message['MessageId']}"
                    ).run(task, **inputs)
                    log.info("launched %s", run.name)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        log.exception("launch failed")
                        continue  # leave the message; SQS redelivers it
            # Deleting is the acknowledgement, and it happens once the run exists.
            sqs.delete_message(
                QueueUrl=QUEUE_URL, ReceiptHandle=message["ReceiptHandle"]
            )
# {{/docs-fragment sqs_poll}}


@app.get("/health")
def health():
    return ({"status": "ok"}, 200) if _alive else ({"status": "starting"}, 503)


# {{docs-fragment sqs_app}}
app_env = FastAPIAppEnvironment(
    name="sqs-subscriber",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "boto3", "fastapi", "uvicorn"
    ),
    # Apps scale to zero by default and autoscale on request volume. A subscriber
    # serves no requests, so without this it would be scaled away.
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    secrets=[flyte.Secret("flyte-api-key", as_env_var="FLYTE_API_KEY")],
    env_vars={
        "QUEUE_URL": "<your queue url>",
        "FLYTE_TASK": "event_driven.on_object",
        "FLYTE_RELEASE": "prod",
    },
)


@app_env.on_startup
async def start() -> None:
    flyte.init_from_api_key(
        project=flyte.current_project(), domain=flyte.current_domain()
    )
    threading.Thread(target=poll_forever, daemon=True).start()
# {{/docs-fragment sqs_app}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.deploy(app_env))
