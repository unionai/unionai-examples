"""Shared configuration for the speculative decoding tutorial.

Every module in this directory imports its models, images, and task environments from
here, so the whole pipeline changes hardware or package versions in one place.
"""

import flyte

# {{docs-fragment models}}
# The model you serve (the verifier) and the public EAGLE-3 draft head you start from.
TARGET_MODEL = "Qwen/Qwen3-8B"
PUBLIC_DRAFTER = "RedHatAI/Qwen3-8B-speculator.eagle3"

# Number of tokens the draft head proposes per verification step.
NUM_SPECULATIVE_TOKENS = 3

# One GPU type for every GPU task. Qwen3-8B in bf16 needs about 17 GB, so any 40 GB+ card
# works. On H100 or H200, change this one line.
GPU = "L40s"
# {{/docs-fragment models}}

VLLM_VERSION = "0.33.0"
SPECULATORS_VERSION = "0.9.0"
FLASHINFER_VERSION = "0.7.2.post1"

# {{docs-fragment images}}
# One GPU image runs vLLM (serving, data generation, benchmarks) and speculators (training).
# vLLM 0.33 and speculators 0.9 agree on torch 2.14 and transformers 5.16-5.19.
gpu_image = (
    flyte.Image.from_debian_base(name="specdec-gpu", python_version=(3, 12))
    .with_apt_packages("curl", "git")
    .with_pip_packages("flyteplugins-vllm==2.13.0", "unionai-reuse>=0.1.3")
    .with_pip_packages(
        f"vllm=={VLLM_VERSION}",
        f"speculators=={SPECULATORS_VERSION}",
        "transformers==5.19.0",
        "arctic-inference==0.3.0",
        "datasets>=4.0.0",
        "httpx>=0.27",
    )
    # Prebuilt FlashInfer kernels, so vLLM's sampler does not JIT-compile at startup.
    .with_pip_packages(
        f"flashinfer-jit-cache=={FLASHINFER_VERSION}",
        index_url="https://flashinfer.ai/whl/cu130",
    )
)

# Orchestration, data wrangling, and reports run on a small CPU image.
cpu_image = flyte.Image.from_debian_base(name="specdec-cpu", python_version=(3, 12)).with_pip_packages(
    "datasets>=4.0.0", "httpx>=0.27"
)
# {{/docs-fragment images}}


# {{docs-fragment resources}}
# Requests sized to the node shapes they land on. The defaults fit AWS L40S nodes:
# g6e.2xlarge (1 GPU, ~7.9 allocatable vCPU, ~61 GiB) and g6e.12xlarge (4 GPUs, 48 vCPU,
# 384 GiB). CPU-only tasks fit a t3a.xlarge (~3.9 vCPU, ~14.5 GiB). Leave headroom below a
# node's allocatable capacity, or the pod never schedules.
ONE_GPU = flyte.Resources(cpu=6, memory="48Gi", gpu=f"{GPU}:1", disk="100Gi", shm="auto")
FOUR_GPUS = flyte.Resources(cpu=40, memory="320Gi", gpu=f"{GPU}:4", disk="200Gi", shm="auto")

# Training needs a vLLM server for hidden states and the trainer. With 4 GPUs, two serve and
# two train. With 1, both share one GPU: slower, but it fits a single 48 GB card.
TRAIN_NUM_GPUS = 1
# {{/docs-fragment resources}}


# {{docs-fragment envs}}
data_env = flyte.TaskEnvironment(
    name="specdec-data",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="12Gi", disk="20Gi"),
)

# Data generation keeps one vLLM server warm per replica, so each shard skips the model load.
generate_env = flyte.TaskEnvironment(
    name="specdec-generate",
    image=gpu_image,
    resources=ONE_GPU,
    reusable=flyte.ReusePolicy(replicas=(1, 4), concurrency=1, idle_ttl=600, scaledown_ttl=300),
)

# Online training: vLLM serves hidden states while the trainer consumes them.
train_env = flyte.TaskEnvironment(
    name="specdec-train",
    image=gpu_image,
    resources=FOUR_GPUS if TRAIN_NUM_GPUS == 4 else ONE_GPU,
)

# Each benchmark task owns one GPU and one vLLM server configuration.
bench_env = flyte.TaskEnvironment(
    name="specdec-bench",
    image=gpu_image,
    resources=ONE_GPU,
)

# The driver only orchestrates. depends_on registers every environment it calls.
driver_env = flyte.TaskEnvironment(
    name="specdec-driver",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[data_env, generate_env, train_env, bench_env],
)
# {{/docs-fragment envs}}
