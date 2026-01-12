# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b45",
#    "flyteplugins-vllm==2.0.0b45",
# ]
# ///

from flyteplugins.vllm import VLLMAppEnvironment
import flyte

# {{docs-fragment vllm-autoscaling}}
vllm_app = VLLMAppEnvironment(
    name="llm-serving-app",
    model_hf_path="Qwen/Qwen3-0.6B",
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1"),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),  # Scale to zero when idle
        scaledown_after=600,  # 10 minutes idle before scaling down
    ),
    requires_auth=False,
)
# {{/docs-fragment vllm-autoscaling}}

