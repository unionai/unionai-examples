# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0",
#    "flyteplugins-slurm"
# ]
# main = "train"
# params = "--steps 500"
# ///

# # Slurm connector - a Python task on an HPC cluster
#
# A native `slurm` task submits the task's *own* container image and the Flyte entrypoint
# as an sbatch job. Inputs, outputs, caching and retries behave exactly as they do for a
# Kubernetes pod: delete `plugin_config` and this runs on Kubernetes unchanged, which also
# makes it the quickest way to tell a Slurm problem from a task problem.
#
# The cluster needs a container runtime on its compute nodes - Pyxis/Enroot by default, or
# Apptainer - and the image must be pullable from there.

import os
import pathlib

from flyteplugins.slurm import Slurm

import flyte
from flyte.io import File

image = flyte.Image.from_debian_base()

env = flyte.TaskEnvironment(
    name="slurm-native",
    image=image,
    plugin_config=Slurm(
        partition="main",
        nodes=1,
        cpus_per_task=4,
        mem="8G",
        time_limit="1:00:00",
        # To request GPUs, add `gres="gpu:1"` or `gpus_per_node=1` - but only if the
        # cluster declares GRES. Without it, sbatch rejects the job outright with
        # `Invalid generic resource (gres) specification`. Check `sinfo -N -o "%N %G"`.
        #
        # Pyxis/Enroot launches the image by default. For an Apptainer cluster:
        #   container_runtime="apptainer",
        #   container_args=["--nv"],          # required for GPUs under Apptainer
        #   modules=["apptainer"],            # if the binary comes from an env module
        #
        # The job reads its inputs and writes its outputs from inside the container, so
        # the node needs credentials for the run's object storage. Mount them from the
        # cluster's shared filesystem and point at the path - never put a secret in `env`,
        # which is rendered into the sbatch script in plain text. Set whichever variable
        # your store reads:
        #   AWS_SHARED_CREDENTIALS_FILE     S3 and S3-compatible (MinIO, R2, Ceph, ...)
        #   GOOGLE_APPLICATION_CREDENTIALS  Google Cloud Storage
        #   AZURE_STORAGE_*                 Azure Blob
        container_mounts=["/home/flyte/.cloud:/etc/cloud:ro"],
        env={"AWS_SHARED_CREDENTIALS_FILE": "/etc/cloud/credentials"},
        # Anything sbatch accepts that is not a first-class field.
        sbatch_options={"requeue": True},
        # Connection details, or leave them out and set FLYTE_SLURM_HOST /
        # FLYTE_SLURM_USERNAME / FLYTE_SLURM_SSH_PRIVATE_KEY on the connector deployment,
        # which is the usual arrangement - the connector's environment wins over these.
        # host="login.hpc.example.com",
        # username="flyte",
        # ssh_private_key="slurm-ssh-key",   # the name of a Flyte secret
    ),
    # Do NOT set `resources` here: the allocation is described by the Slurm fields above
    # and granted by Slurm, not by Kubernetes.
)


# `cache` and `retries` are per-task, not per-environment. PREEMPTED maps to
# RETRYABLE_FAILED, so a preempted allocation is retried rather than failing the run.
@env.task(cache="auto", retries=2)
async def train(steps: int = 100) -> File:
    """Write a 'model' to the run's object storage and return a reference to it."""
    out = pathlib.Path("/tmp/model.txt")
    out.write_text(f"trained for {steps} steps on {os.uname().nodename}\n")
    # File.from_local uploads to the run's raw-data path, so any task - on Slurm or on
    # Kubernetes - can read it afterwards.
    return await File.from_local(out)


@env.task
async def where_am_i() -> str:
    """Smallest possible proof the task body really ran on a Slurm worker."""
    return f"{os.uname().nodename} (SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', 'unset')})"


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train, steps=500)
    print(run.url)
    run.wait()
