# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-vllm>=2.0.0b45",
# ]
# ///

"""vLLM app with multi-GPU tensor parallelism."""

from flyteplugins.vllm import VLLMAppEnvironment
import flyte

# {{docs-fragment multi-gpu}}
vllm_app = VLLMAppEnvironment(
    name="multi-gpu-llm-app",
    model_hf_path="meta-llama/Llama-2-70b-hf",
    model_id="llama-2-70b",
    resources=flyte.Resources(
        cpu="8",
        memory="32Gi",
        gpu="L40s:4",  # 4 GPUs for tensor parallelism
        disk="100Gi",
    ),
    extra_args=[
        "--tensor-parallel-size", "4",  # Use 4 GPUs
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9",
    ],
    requires_auth=False,
)
# {{/docs-fragment multi-gpu}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
# {{/docs-fragment deploy}}
