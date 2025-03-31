from flytekit import Artifact, ImageSpec, PodTemplate, task, Resources
from typing import Annotated
from kubernetes.client.models import (
    V1Container,
    V1PodSpec,
    V1ResourceRequirements,
    V1VolumeMount,
    V1Volume,
    V1SecurityContext,
    V1Probe,
    V1HTTPGetAction,
)
from flytekit.types.directory import FlyteDirectory


MoonDreamArtifact = Artifact(name="ollama-moondream")
image = ImageSpec(
    name="ollama-serve",
    packages=["ollama==0.4.4", "kubernetes==31.0.0"],
    registry="ghcr.io/unionai-oss",
)


template = PodTemplate(
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                image=image,
                volume_mounts=[
                    V1VolumeMount(name="ollama-cache", mount_path="/root/.ollama")
                ],
                security_context=V1SecurityContext(
                    run_as_user=0,
                ),
            ),
        ],
        init_containers=[
            V1Container(
                name="ollama",
                image="ollama/ollama:0.6.3",
                resources=V1ResourceRequirements(
                    requests={"cpu": "2", "memory": "6Gi"},
                    limits={"cpu": "2", "memory": "6Gi"},
                ),
                restart_policy="Always",
                volume_mounts=[
                    V1VolumeMount(name="ollama-cache", mount_path="/root/.ollama")
                ],
                startup_probe=V1Probe(http_get=V1HTTPGetAction(path="/", port=11434)),
            ),
        ],
        volumes=[V1Volume(name="ollama-cache", empty_dir={})],
    )
)


@task(pod_template=template, limits=Resources(cpu="1", mem="4Gi"))
def download_model(
    model: str = "moondream",
) -> Annotated[FlyteDirectory, MoonDreamArtifact]:
    import ollama

    ollama.pull(model)
    return FlyteDirectory(path="/root/.ollama")
