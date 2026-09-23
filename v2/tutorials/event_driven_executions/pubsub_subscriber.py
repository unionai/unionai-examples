# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte>=2.6.12",
#     "google-cloud-pubsub",
#     "fastapi",
#     "uvicorn",
# ]
# ///
"""Read a Pub/Sub subscription and launch a run per message, as a Union app."""

import logging
import os
from datetime import datetime

import flyte
import flyte.remote as remote
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from google.cloud import pubsub_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pubsub")

TASK = os.environ["FLYTE_TASK"]
RELEASE = os.environ.get("FLYTE_RELEASE", "prod")
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "20"))

app = FastAPI()
_future = None


# {{docs-fragment pubsub_inputs}}
def to_inputs(message) -> dict | None:
    """Map a GCS notification to task inputs, or None if there is nothing to act on.

    GCS puts routing data in `attributes` and the full object resource in `data`.
    That resource describes the file — `kind`, `selfLink`, `md5Hash` and so on — so
    it is metadata rather than task inputs, and reading `attributes` is both simpler
    and more stable.
    """
    attrs = message.attributes
    bucket, obj = attrs.get("bucketId"), attrs.get("objectId")
    if not bucket or not obj:
        return None
    if attrs.get("eventType") != "OBJECT_FINALIZE":
        return None  # deletes and metadata updates arrive here too

    inputs = {"object_key": f"gs://{bucket}/{obj}"}
    if timestamp := attrs.get("eventTime"):
        inputs["event_time"] = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return inputs
# {{/docs-fragment pubsub_inputs}}


# {{docs-fragment pubsub_handle}}
def handle(message) -> None:
    inputs = to_inputs(message)
    if inputs is None:
        message.ack()  # nothing to do; ack so it is not redelivered
        return

    try:
        # The run name comes from the Pub/Sub messageId, so a redelivery collides
        # with the run that already exists rather than starting a second one.
        task = remote.Task.get(TASK, version=RELEASE)
        run = flyte.with_runcontext(name=f"ps-{message.message_id}").run(task, **inputs)
        log.info("launched %s", run.name)
    except Exception as e:
        if "already exists" not in str(e).lower():
            log.exception("launch failed")
            message.nack()  # real failure; let Pub/Sub redeliver
            return

    message.ack()  # acknowledge once the run exists, never on completion
# {{/docs-fragment pubsub_handle}}


@app.get("/health")
def health():
    running = _future is not None and _future.running()
    return ({"status": "ok"}, 200) if running else ({"status": "stream not running"}, 503)


# {{docs-fragment pubsub_app}}
app_env = FastAPIAppEnvironment(
    name="pubsub-subscriber",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "google-cloud-pubsub", "fastapi", "uvicorn"
    ),
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    secrets=[flyte.Secret("flyte-api-key", as_env_var="FLYTE_API_KEY")],
    env_vars={
        "GCP_PROJECT": "<your gcp project>",
        "SUBSCRIPTION": "<your subscription>",
        "FLYTE_TASK": "event_driven.on_object",
        "FLYTE_RELEASE": "prod",
    },
)


@app_env.on_startup
async def start() -> None:
    global _future
    flyte.init_from_api_key(
        project=flyte.current_project(), domain=flyte.current_domain()
    )
    subscriber = pubsub_v1.SubscriberClient()
    path = subscriber.subscription_path(
        os.environ["GCP_PROJECT"], os.environ["SUBSCRIPTION"]
    )
    # max_messages bounds how much work is in flight.
    _future = subscriber.subscribe(
        path,
        callback=handle,
        flow_control=pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES),
    )
# {{/docs-fragment pubsub_app}}

# This app is deployed from the CLI rather than from a `__main__` guard:
#
#     flyte deploy pubsub_subscriber.py app_env
#
# It has no task to invoke, and it needs a live subscription plus cloud
# credentials, so there is nothing for the example harness to run locally.
