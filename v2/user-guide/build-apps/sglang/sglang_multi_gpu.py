# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-sglang>=2.0.0b45",
# ]
# ///

"""SGLang app with multi-GPU tensor parallelism."""

from flyteplugins.sglang import SGLangAppEnvironment
import flyte


# {{docs-fragment multi-gpu}}
sglang_app = SGLangAppEnvironment(
    name="multi-gpu-sglang-app",
    model_hf_path="meta-llama/Llama-2-70b-hf",
    model_id="llama-2-70b",
    resources=flyte.Resources(
        cpu="8",
        memory="32Gi",
        gpu="L40s:4",  # 4 GPUs for tensor parallelism
        disk="100Gi",
    ),
    extra_args=[
        "--tp", "4",  # Tensor parallelism size (4 GPUs)
        "--max-model-len", "4096",
        "--mem-fraction-static", "0.9",
    ],
    requires_auth=False,
)
# {{/docs-fragment multi-gpu}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(sglang_app)
    print(f"Deployed SGLang app: {app.url}")
# {{/docs-fragment deploy}}
