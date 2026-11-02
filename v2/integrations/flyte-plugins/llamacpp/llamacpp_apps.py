# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "flyteplugins-llamacpp>=2.12.2",
# ]
# ///

"""Serve GGUF models with llama.cpp."""

import flyte
import flyte.app
from flyteplugins.llamacpp import LlamaCppAppEnvironment, build_llama_cpp_image

# {{docs-fragment object-storage}}
llama_app = LlamaCppAppEnvironment(
    name="my-llm-app",
    # A directory of GGUF weights, or a direct path to one .gguf file.
    model_path="s3://<your-bucket>/models/<your-model-gguf>",
    model_id="my-model",
    resources=flyte.Resources(cpu="4", memory="32Gi", gpu="L40s:1", disk="100Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)
# {{/docs-fragment object-storage}}

# {{docs-fragment hugging-face}}
gemma_app = LlamaCppAppEnvironment(
    name="gemma-app",
    model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    model_id="gemma-3-4b-it",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="50Gi"),
)
# {{/docs-fragment hugging-face}}

# {{docs-fragment custom-image}}
fat_binary_app = LlamaCppAppEnvironment(
    name="fat-binary-app",
    model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    model_id="gemma-3-4b-it",
    image=build_llama_cpp_image(
        cuda_arch="80;86;89;90",  # A100, A10, L4 and L40S, H100
        ref="b6148",  # A llama.cpp release tag
    ),
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="50Gi"),
)
# {{/docs-fragment custom-image}}

# {{docs-fragment cpu-only}}
cpu_app = LlamaCppAppEnvironment(
    name="cpu-app",
    model_hf_path="ggml-org/Qwen3-0.6B-GGUF:Q8_0",
    model_id="qwen3-0.6b",
    image=build_llama_cpp_image(cuda=False),
    resources=flyte.Resources(cpu="4", memory="8Gi", disk="10Gi"),
)
# {{/docs-fragment cpu-only}}

# {{docs-fragment extra-args}}
tuned_app = LlamaCppAppEnvironment(
    name="tuned-app",
    model_hf_path="ggml-org/gemma-3-4b-it-GGUF:Q4_K_M",
    model_id="gemma-3-4b-it",
    extra_args="--ctx-size 32768 --parallel 4 --jinja",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="50Gi"),
)
# {{/docs-fragment extra-args}}

# {{docs-fragment speculative}}
speculative_app = LlamaCppAppEnvironment(
    name="speculative-app",
    model_path="s3://<your-bucket>/models/qwen3-32b-gguf",
    model_id="qwen3-32b",
    draft_model_hf_path="ggml-org/Qwen3-0.6B-GGUF:Q8_0",
    extra_args="--draft-max 16 --draft-min 1 --gpu-layers-draft 99",
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu="L40s:1", disk="120Gi"),
)
# {{/docs-fragment speculative}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(gemma_app)
    print(f"Deployed llama.cpp app: {app.url}")
# {{/docs-fragment deploy}}
