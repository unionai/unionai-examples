"""SGLang app using prefetched models."""

from flyteplugins.sglang import SGLangAppEnvironment
import flyte

# {{docs-fragment prefetch}}
# Prefetch the model first
run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
run.wait()

# Use the prefetched model
sglang_app = SGLangAppEnvironment(
    name="my-sglang-app",
    model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1", disk="10Gi"),
    stream_model=True,  # Stream model directly from blob store to GPU
    requires_auth=False,
)
# {{/docs-fragment prefetch}}

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(sglang_app)
    print(f"Deployed SGLang app: {app.url}")

