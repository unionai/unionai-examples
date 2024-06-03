from datetime import datetime, timedelta, timezone
from typing import Tuple

from flytekit import task
from flytekit.core.artifact import Artifact
from flytekit.types.file import FlyteFile
from typing_extensions import Annotated


HOUR_CYCLE = 6
ImageDir = Artifact(name="image_directory")


@task
def fetch_images(
    images_response: dict, kickoff_time: datetime
) -> Tuple[int, list[FlyteFile]]:
    images = []
    start_time = kickoff_time - timedelta(hours=HOUR_CYCLE)

    n_images = 0
    if "Contents" in images_response:
        for obj in images_response["Contents"]:
            if obj["LastModified"].replace(tzinfo=timezone.utc) >= start_time:
                if not obj["Key"].endswith("/"):
                    images.append(f"s3://{images_response["Name"]}/{obj['Key']}")
                    n_images += 1

    return n_images, images


@task
def emit_artifact(img_dir: list[FlyteFile]) -> Annotated[list[FlyteFile], ImageDir]:
    print("emit")
    return ImageDir.create_from(img_dir)


@task
def dont_emit_artifact():
    print("dont_emit")
