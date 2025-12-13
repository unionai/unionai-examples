"""vLLM app using prefetched models."""

from flyteplugins.vllm import VLLMAppEnvironment
import flyte

# {{docs-fragment prefetch}}
# Prefetch the model first
run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
run.wait()

# Use the prefetched model
vllm_app = VLLMAppEnvironment(
    name="my-llm-app",
    model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1", disk="10Gi"),
    stream_model=True,  # Stream model directly from blob store to GPU
    requires_auth=False,
)
# {{/docs-fragment prefetch}}

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(vllm_app)
    print(f"Deployed vLLM app: {app.url}")

