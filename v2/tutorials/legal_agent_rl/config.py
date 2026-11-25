"""Where things run, and with what. One program, two clusters.

``CLUSTER=demo``  a single NVIDIA GPU (L40s by default) on a regular Union cluster.
``CLUSTER=amd``   a whole 8x AMD MI355X node from a GPU market, reached through a queue (``GPU_QUEUE``).

A whole-node GPU pool admits pods that ask for all eight GPUs, in a pod template, with no node selector (the queue
adds it). Everything in a run has to live in the same cluster pool, so on ``amd`` the CPU driver goes to the queue too.
"""

from __future__ import annotations

import os

import flyte

from images import CPU_IMAGE, CUDA_IMAGE, ROCM_IMAGE

CLUSTER = os.getenv("CLUSTER", "demo")
QUEUE = os.getenv("GPU_QUEUE") or None
GPUS_PER_NODE = 8 if CLUSTER == "amd" else 1
# Pin prebuilt image URIs (python build.py prints them). A pool's builder can't always push, so build off-queue first.
GPU_IMAGE_URI = os.getenv("GPU_IMAGE_URI")
CPU_IMAGE_URI = os.getenv("CPU_IMAGE_URI")


def whole_node_pod_template() -> flyte.PodTemplate:
    from kubernetes.client import V1Container, V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector, V1PodSpec, \
        V1ResourceRequirements

    gpus = {"amd.com/gpu": str(GPUS_PER_NODE)}
    node = V1EnvVar(name="NODE_NAME", value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(field_path="spec.nodeName")))
    return flyte.PodTemplate(primary_container_name="primary", pod_spec=V1PodSpec(containers=[
        V1Container(name="primary", resources=V1ResourceRequirements(requests=gpus, limits=gpus), env=[node])]))


def gpu_env_kwargs() -> dict:
    if CLUSTER == "amd":
        return dict(image=flyte.Image.from_base(GPU_IMAGE_URI) if GPU_IMAGE_URI else ROCM_IMAGE,
                    resources=flyte.Resources(cpu=96, memory="768Gi", disk="800Gi", shm="auto"),
                    pod_template=whole_node_pod_template(), queue=QUEUE)
    gpu = os.getenv("DEMO_GPU", "L40s")
    return dict(image=flyte.Image.from_base(GPU_IMAGE_URI) if GPU_IMAGE_URI else CUDA_IMAGE,
                resources=flyte.Resources(cpu=8, memory="64Gi", gpu=f"{gpu}:1", disk="200Gi", shm="auto"))


def cpu_env_kwargs(cpu: int = 2, memory: str = "4Gi") -> dict:
    return dict(image=flyte.Image.from_base(CPU_IMAGE_URI) if CPU_IMAGE_URI else CPU_IMAGE,
                resources=flyte.Resources(cpu=cpu, memory=memory), queue=QUEUE)


# The variables a child task's spec depends on are read in the parent's container, so forward them.
PROPAGATE = {k: v for k in ("CLUSTER", "GPU_QUEUE", "GPU_IMAGE_URI", "CPU_IMAGE_URI", "DEMO_GPU", "HF_HOME")
             if (v := os.getenv(k))}
PROPAGATE.setdefault("HF_HOME", "/tmp/hf")
