# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "flyteplugins-vllm>=2.0.0b45",
# ]
# ///

"""A simple vLLM app example."""

from flyteplugins.vllm import VLLMAppEnvironment
import flyte

# {{docs-fragment basic-vllm-app}}
vllm_app = VLLMAppEnvironment(
    name="my-llm-app",
    model_hf_path="Qwen/Qwen3-0.6B",  # HuggingFace model path
    model_id="qwen3-0.6b",  # Model ID exposed by vLLM
    resources=flyte.Resources(
        cpu="4",
        memory="16Gi",
        gpu="L40s:1",  # GPU required for LLM serving
        disk="10Gi",
    ),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
)
# {{/docs-fragment basic-vllm-app}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
# {{/docs-fragment deploy}}
