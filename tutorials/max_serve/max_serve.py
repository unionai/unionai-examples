# # Serve your LLM with MAX Serve
#
# [MAX Serve](https://docs.modular.com/max/api/serve/) is a high-performance inference server for deploying
# large language models. In this tutorial, we learn how to cache a model from HuggingFace and serve with
# MAX Serve and Union Serving.
#
# {{run-on-union}}
#
# ## Managing Dependencies
#
# First we import the dependencies for defining the Union App:

from union import Resources, ImageSpec, Artifact
from union.app import App, Input
from flytekit.extras.accelerators import L4
import os

# For defining the image, we install `union-runtime` into Modular's base name with the ImageSpec
# image builder. Set the `IMAGE_SPEC_REGISTRY` environment variable to be a public registry you can push to.
# With `python_exec="/opt/venv/bin/python"`, we configure the image builder to install any new packages
# into the base image's python environment.

image = ImageSpec(
    name="modular-max",
    base_image="modular/max-nvidia-base:25.4.0.dev2025050705",
    builder="default",
    packages=["union-runtime>=0.1.18"],
    entrypoint=["/bin/bash"],
    python_exec="/opt/venv/bin/python",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)

# ## Defining the Union App
#
# The workflow in `cache_model.py` caches the Qwen2.5 model from HuggingFace into a Union Artifact. Here
# we use the same Artifact as an Input to the Union App, which gets downloaded in `mount=/root/qwen-0-5`.
# The `args` is set to a **Max Serve** specific entrypoint, where `--model-path=/root/qwen-0-5`
# configures **Max Serve** to load the model from `/root/qwen-0-5`.

Qwen_Coder_Artifact = Artifact(name="Qwen2.5-Coder-0.5B")
modular_model = App(
    name="modular-qwen-0-5-coder",
    container_image=image,
    inputs=[Input(name="model", value=Qwen_Coder_Artifact.query(), env_var="MODEL", mount="/root/qwen-0-5")],
    args=[
        "python",
        "-m",
        "max.entrypoints.pipelines",
        "serve",
        "--model-path=/root/qwen-0-5",
        "--device-memory-utilization",
        "0.7",
        "--max-length",
        "2048",
    ],
    port=8000,
    requests=Resources(cpu="7", mem="20Gi", gpu="1", ephemeral_storage="20Gi"),
    accelerator=L4,
    scaledown_after=300,
)

# ## Caching and deploying The App
#
# Run the workflow to cache the LLM:
#
# ```bash
# union run --remote cache_model.py cache_model
# ```
#
# Deploy the Union App backed by **Max Serve**:
# ```bash
# union deploy apps max_serve.py modular-qwen-0-5-coder
# ```
