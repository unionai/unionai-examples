# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "flyteplugins-vllm>=2.0.0b45",
# ]
# override-dependencies = [
#    "cel-python; sys_platform == 'never'",
#]
# ///

"""vLLM app using prefetched models."""

from flyteplugins.vllm import VLLMAppEnvironment
import flyte


# {{docs-fragment prefetch}}

# Use the prefetched model
vllm_app = VLLMAppEnvironment(
    name="my-llm-app",
    model_hf_path="Qwen/Qwen3-0.6B",  # this is a placeholder
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1", disk="10Gi"),
    stream_model=True,  # Stream model directly from blob store to GPU
    requires_auth=False,
)

if __name__ == "__main__":
    flyte.init_from_config()

    # Prefetch the model first
    run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
    run.wait()

    # Use the prefetched model
    app = flyte.serve(
        vllm_app.clone_with(
            vllm_app.name,
            model_hf_path=None,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
        )
    )
    print(f"Deployed vLLM app: {app.url}")
# {{/docs-fragment prefetch}}