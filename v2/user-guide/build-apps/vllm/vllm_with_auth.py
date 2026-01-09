# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "flyteplugins-vllm>=2.0.0b45",
# ]
# ///

"""vLLM app with API key authentication."""

import pathlib
from flyteplugins.vllm import VLLMAppEnvironment
import flyte


# The secret must be created using: flyte create secret AUTH_SECRET <your-api-key-value>
vllm_app = VLLMAppEnvironment(
    name="vllm-app-with-auth",
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
    # Disable Union's platform-level authentication so you can access the
    # endpoint from the public internet
    requires_auth=False,
    # Inject the secret as an environment variable
    secrets=flyte.Secret(key="AUTH_SECRET", as_env_var="AUTH_SECRET"),
    # Pass the API key to vLLM's --api-key argument
    # The $AUTH_SECRET will be replaced with the actual secret value at runtime
    extra_args=[
        "--api-key", "$AUTH_SECRET",
    ],
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")
