# /// script
# requires-python = "==3.12"
# dependencies = [
#     "flyte>=2.6.12",
# ]
# ///
"""The same automation without a queue: publish an artifact, and a trigger fires."""

from datetime import datetime, timezone

import flyte
from flyte.io import File
from flyte.remote import Artifact

env = flyte.TaskEnvironment(name="artifact_chain")

ARTIFACT = "incoming_dataset"


# {{docs-fragment artifact_trigger}}
# Fires on every new version of the artifact. `TriggeredArtifact` binds the artifact
# that fired the trigger to a task input, the way `TriggerTime` does for a schedule.
on_new_data = flyte.Trigger(
    name="on_new_dataset",
    automation=flyte.OnArtifact(name=ARTIFACT),
    inputs={"dataset": flyte.TriggeredArtifact},
)


@env.task(triggers=[on_new_data])
async def consume(dataset: File) -> str:
    """Runs automatically whenever a new version of the artifact is published."""
    return f"consumed {dataset.path} on task version {flyte.ctx().version}"
# {{/docs-fragment artifact_trigger}}


# {{docs-fragment artifact_produce}}
@env.task
async def produce(rows: int = 3) -> str:
    """Write a file and publish it as a new version of the artifact."""
    path = "/tmp/dataset.csv"
    with open(path, "w") as fh:
        fh.write("id,value\n")
        for i in range(rows):
            fh.write(f"{i},{i * 10}\n")

    file = await File.from_local(path)

    # `external_ref` is required when publishing from inside a task. Without it the
    # SDK derives provenance from the running action but omits the org, project and
    # domain, and the request is rejected.
    artifact = await Artifact.create.aio(
        file,
        name=ARTIFACT,
        description="Dataset produced by the upstream task",
        external_ref=file.path,
        attrs={"rows": str(rows), "at": datetime.now(timezone.utc).isoformat()},
    )
    return f"published {artifact.name}:{artifact.version}"
# {{/docs-fragment artifact_produce}}


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.deploy(env))
