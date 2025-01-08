import os
from kubernetes.client.models import (
    V1Container,
    V1PodSpec,
    V1ResourceRequirements,
    V1EnvVar,
)
import union

image = union.ImageSpec(
    registry=os.environ.get("DOCKER_REGISTRY", None),
    packages=["union", "flytekitplugins-pod"],
)

pod_template = union.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                image=image,
                resources=V1ResourceRequirements(
                    requests={
                        "cpu": "1",
                        "memory": "1Gi",
                    },
                    limits={
                        "cpu": "1",
                        "memory": "1Gi",
                    },
                ),
                env=[V1EnvVar(name="COMP_KEY_EX", value="compile_time")],
            ),
        ],
    ),
)

actor = union.ActorEnvironment(
    name="my-actor",
    replica_count=1,
    ttl_seconds=30,
    pod_template=pod_template,
)

@actor.task
def get_and_set() -> str:
    os.environ["RUN_KEY_EX"] = "run_time"
    return os.getenv("COMP_KEY_EX")


@actor.task
def check_set() -> str:
    return os.getenv("RUN_KEY_EX")


@union.workflow
def wf() -> tuple[str,str]:
    return get_and_set(), check_set()
