from flytekit import Email, LaunchPlan, Secret, WorkflowExecutionPhase, workflow
from flytekit.types.file import FlyteFile
from flytekitplugins.openai import BatchResult, create_batch
from union.artifacts import OnArtifact

from .fetch_images import ImageDir
from .make_requests import create_request

YOUR_EMAIL = "your@email.com"
SECRET_GROUP = "arn:aws:secretsmanager:us-east-2:356633062068:secret:"
SECRET_KEY = "daniel-open-ai-key-iwi4Wv"


on_image_dir = OnArtifact(trigger_on=ImageDir, inputs={"img_dir": ImageDir})


file_batch = create_batch(
    name="image-moderation",
    openai_organization="unionai",
    secret=Secret(
        group=SECRET_GROUP, key=SECRET_KEY, mount_requirement=Secret.MountType.FILE
    ),
)


@workflow
def batch_wf(img_dir: list[FlyteFile]) -> BatchResult:
    json_generator = create_request(img_dir=img_dir)
    return file_batch(jsonl_in=json_generator)


openai_lp = LaunchPlan.get_or_create(
    name="openai_lp",
    workflow=batch_wf,
    trigger=on_image_dir,
    notifications=[
        Email(
            phases=[WorkflowExecutionPhase.SUCCEEDED],
            recipients_email=[YOUR_EMAIL],
        )
    ],
)
