"""A simple SGLang app example."""

from flyteplugins.sglang import SGLangAppEnvironment
import flyte

# {{docs-fragment basic-sglang-app}}
sglang_app = SGLangAppEnvironment(
    name="my-sglang-app",
    model_hf_path="Qwen/Qwen3-0.6B",  # HuggingFace model path
    model_id="qwen3-0.6b",  # Model ID exposed by SGLang
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
# {{/docs-fragment basic-sglang-app}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(sglang_app)
    print(f"Deployed SGLang app: {app.url}")
# {{/docs-fragment deploy}}
