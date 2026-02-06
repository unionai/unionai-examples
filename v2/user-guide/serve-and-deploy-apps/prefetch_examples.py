# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-vllm>=2.0.0b49",
# ]
# ///

"""Prefetch examples for the prefetching-models.md documentation."""

import flyte
from flyte.prefetch import ShardConfig, VLLMShardArgs
from flyteplugins.vllm import VLLMAppEnvironment


# {{docs-fragment basic-prefetch}}
# Prefetch a HuggingFace model
run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")

# Wait for prefetch to complete
run.wait()

# Get the model path
model_path = run.outputs()[0].path
print(f"Model prefetched to: {model_path}")
# {{/docs-fragment basic-prefetch}}


# {{docs-fragment using-prefetched-models}}
# Prefetch the model
run = flyte.prefetch.hf_model(repo="Qwen/Qwen3-0.6B")
run.wait()

# Use the prefetched model
vllm_app = VLLMAppEnvironment(
    name="my-llm-app",
    model_path=flyte.app.RunOutput(
        type="directory",
        run_name=run.name,
    ),
    model_id="qwen3-0.6b",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1"),
    stream_model=True,
)

app = flyte.serve(vllm_app)
# {{/docs-fragment using-prefetched-models}}


# {{docs-fragment custom-artifact-name}}
run = flyte.prefetch.hf_model(
    repo="Qwen/Qwen3-0.6B",
    artifact_name="qwen-0.6b-model",  # Custom name for the stored model
)
# {{/docs-fragment custom-artifact-name}}


# {{docs-fragment hf-token}}
run = flyte.prefetch.hf_model(
    repo="meta-llama/Llama-2-7b-hf",
    hf_token_key="HF_TOKEN",  # Name of Flyte secret containing HF token
)
# {{/docs-fragment hf-token}}


# {{docs-fragment with-resources}}
run = flyte.prefetch.hf_model(
    repo="Qwen/Qwen3-0.6B",
    cpu="4",
    mem="16Gi",
    ephemeral_storage="100Gi",
)
# {{/docs-fragment with-resources}}


# {{docs-fragment vllm-sharding}}
run = flyte.prefetch.hf_model(
    repo="meta-llama/Llama-2-70b-hf",
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu="L40s:4"),
    shard_config=ShardConfig(
        engine="vllm",
        args=VLLMShardArgs(
            tensor_parallel_size=4,
            dtype="auto",
            trust_remote_code=True,
        ),
    ),
    hf_token_key="HF_TOKEN",
)

run.wait()
# {{/docs-fragment vllm-sharding}}


# {{docs-fragment using-sharded-models}}
# Use in vLLM app
vllm_app = VLLMAppEnvironment(
    name="multi-gpu-llm-app",
    # this will download the model from HuggingFace into the app container's filesystem
    model_hf_path="Qwen/Qwen3-0.6B",
    model_id="llama-2-70b",
    resources=flyte.Resources(
        cpu="8",
        memory="32Gi",
        gpu="L40s:4",  # Match the number of GPUs used for sharding
    ),
    extra_args=[
        "--tensor-parallel-size", "4",  # Match sharding config
    ],
)

if __name__ == "__main__":
    # Prefetch with sharding
    run = flyte.prefetch.hf_model(
        repo="meta-llama/Llama-2-70b-hf",
        accelerator="L40s:4",
        shard_config=ShardConfig(
            engine="vllm",
            args=VLLMShardArgs(tensor_parallel_size=4),
        ),
    )
    run.wait()

    flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            # override the model path to use the prefetched model
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            # set the hf_model_path to None
            hf_model_path=None,
            # stream the model from flyte object store directly to the GPU
            stream_model=True,
        )
    )
# {{/docs-fragment using-sharded-models}}


# {{docs-fragment complete-example}}
# define the app environment
vllm_app = VLLMAppEnvironment(
    name="qwen-serving-app",
    # this will download the model from HuggingFace into the app container's filesystem
    model_hf_path="Qwen/Qwen3-0.6B",
    model_id="qwen3-0.6b",
    resources=flyte.Resources(
        cpu="4",
        memory="16Gi",
        gpu="L40s:1",
        disk="10Gi",
    ),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=600,
    ),
    requires_auth=False,
)

if __name__ == "__main__":
    # prefetch the model
    print("Prefetching model...")
    run = flyte.prefetch.hf_model(
        repo="Qwen/Qwen3-0.6B",
        artifact_name="qwen-0.6b",
        cpu="4",
        mem="16Gi",
        ephemeral_storage="50Gi",
    )

    # wait for completion
    print("Waiting for prefetch to complete...")
    run.wait()
    print(f"Model prefetched: {run.outputs()[0].path}")

    # deploy the app
    print("Deploying app...")
    flyte.init_from_config()
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run.name),
            hf_model_path=None,
            stream_model=True,
        )
    )
    print(f"App deployed: {app.url}")
# {{/docs-fragment complete-example}}
