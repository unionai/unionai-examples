import union
from union.app.llm import VLLMApp
from flytekit.extras.accelerators import A10G

Model = union.Artifact(name="gpt-oss-20b")


image = union.ImageSpec(
    name="vllm-gpt-oss",
    builder="union",
    apt_packages=["build-essential", "wget", "gnupg"],
    packages=[
        "union[vllm]==0.1.191b0",
        "--pre vllm==0.10.1+gptoss \
        --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
        --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
        --index-strategy unsafe-best-match",
    ],
).with_commands(
    [
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-8",
        "/usr/local/cuda/bin/nvcc --version",
        "chown -R union /root",
        "chown -R union /home",
    ]
)


gpt_oss_app = VLLMApp(
    name="gpt-oss-20b-vllm",
    model=Model.query(),
    model_id="gpt-oss",
    container_image=image,
    requests=union.Resources(cpu="5", mem="26Gi", gpu="1", ephemeral_storage="150Gi"),
    accelerator=A10G,
    scaledown_after=300,
    stream_model=False,
    requires_auth=False,
    extra_args="--async-scheduling",
    env={"VLLM_ATTENTION_BACKEND": "TRITON_ATTN_VLLM_V1"},
)
