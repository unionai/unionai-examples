import os
from union.actor import ActorEnvironment
from flytekit import ImageSpec, workflow
from typing import Tuple
from utils import TextSample, TextSampleArtifact

image = ImageSpec(
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    builder="union",
    packages=[
        "union==0.1.82",
    ],
)

actor_env = ActorEnvironment(
    name="vllm-actor",
    replica_count=1,
    container_image=image,
    ttl_seconds=300,
)


@actor_env.task
def ner(text: TextSample) -> Tuple[str, str]:

    response = (text.id, text.body)

    return response


@workflow
def ner_wf(text: TextSample = TextSampleArtifact.query()):
    file_name, named_entities = ner(text=text)
