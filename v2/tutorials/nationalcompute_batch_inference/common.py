"""Shared settings for running on National Compute's AMD MI355X cluster.

National Compute's cluster is its own Union cluster, reached through a queue.
Everything that should run there, model prefetch included, names that queue.
"""

import os
import sys

import flyte

# The queue that routes work to the National Compute cluster.
QUEUE = os.getenv("NC_QUEUE", "national-compute")

# Registry the National Compute nodes pull images from, and the image pull
# secret the builder pushes with. Leave NC_REGISTRY unset to use the builder's
# default registry when the cluster can already pull from it.
REGISTRY = os.getenv("NC_REGISTRY")
REGISTRY_SECRET = os.getenv("NC_REGISTRY_SECRET")

# National Compute's market grants whole nodes, so a GPU pod asks for all eight
# MI355X GPUs.
GPUS_PER_NODE = 8


def with_registry(image: flyte.Image, name: str) -> flyte.Image:
    """Name the image and push it to NC_REGISTRY when one is set."""
    if REGISTRY:
        return image.clone(
            name=name, registry=REGISTRY, registry_secret=REGISTRY_SECRET, extendable=True
        )
    return image.clone(name=name, extendable=True)


def full_node_pod_template() -> flyte.PodTemplate:
    """Request a whole node's AMD GPUs on the task container."""
    from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

    gpus = {"amd.com/gpu": str(GPUS_PER_NODE)}
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    resources=V1ResourceRequirements(requests=gpus, limits=gpus),
                )
            ],
        ),
    )


def prebuild_and_reexec(images: dict[str, flyte.Image]) -> None:
    """Build images without a queue, then rerun this script with their URIs set.

    A run sent to a queue builds its images on that queue's cluster. Building
    first without a queue uses the default builder, which skips images that
    already exist. The script then restarts with each URI in the given
    environment variable, which the task environments read instead of building.
    """
    missing = {var: image for var, image in images.items() if not os.getenv(var)}
    if not missing:
        return
    flyte.init_from_config()
    for var, image in missing.items():
        os.environ[var] = flyte.build(image).uri
        print(f"{var}={os.environ[var]}", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv])
