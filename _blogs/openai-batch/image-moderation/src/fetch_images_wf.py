from datetime import datetime

from flytekit import CronSchedule, LaunchPlan, conditional, kwtypes, workflow
from flytekitplugins.awssagemaker_inference import BotoConfig, BotoTask

from .fetch_images import HOUR_CYCLE, fetch_images, emit_artifact, dont_emit_artifact

YOUR_INPUT_BUCKET = "your_bucket"
YOUR_INPUT_DIRECTORY = "your_directory"

list_files_config = BotoConfig(
    service="s3",
    method="list_objects_v2",
    config={
        "Bucket": "{inputs.bucket}",
        "Prefix": "{inputs.directory}",
    },
    region="us-east-2",
)

list_images_task = BotoTask(
    name="list_images",
    task_config=list_files_config,
    inputs=kwtypes(bucket=str, directory=str),
)


@workflow
def fetch_images_wf(bucket_name: str, directory: str, kickoff_time: datetime):
    images_response = list_images_task(bucket=bucket_name, directory=directory)
    n_images, img_dir = fetch_images(
        images_response=images_response, kickoff_time=kickoff_time
    )
    (
        conditional("check_images")
        .if_(n_images != 0)
        .then(emit_artifact(img_dir=img_dir))
        .else_()
        .then(dont_emit_artifact())
    )


fetch_images_lp = LaunchPlan.get_or_create(
    name="fetch_images_lp",
    workflow=fetch_images_wf,
    schedule=CronSchedule(
        schedule=f"0 */{HOUR_CYCLE} * * *", kickoff_time_input_arg="kickoff_time"
    ),
    default_inputs={
        "bucket_name": YOUR_INPUT_BUCKET,
        "directory": YOUR_INPUT_DIRECTORY,
    },
)
