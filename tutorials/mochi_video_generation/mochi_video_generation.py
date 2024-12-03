# # Video Generation with Mochi
#
# This tutorial demonstrates how to run the Mochi 1 text-to-video generation model by Genmo on Union.
#
# ## Overview
#
# Mochi 1 is an open-source 10-billion parameter diffusion model built on the
# Asymmetric Diffusion Transformer (AsymmDiT) architecture.
# The Mochi model can be run on both single- and multi-GPU setups.
# It is recommended to run the model on an H100 GPU, but quantized versions supported by HuggingFace diffusers
# allow for running the model with a minimum of 22GB VRAM.
#
# Let's begin by importing the necessary dependencies:

from dataclasses import dataclass
from pathlib import Path

import flytekit as fl
from dataclasses_json import dataclass_json
from flytekit import FlyteContextManager
from flytekit.extras.accelerators import A100
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from union.actor import ActorEnvironment

# We also define a dataclass to provide the prompt and the necessary params to be used while generating the videos.


@dataclass_json
@dataclass
class VideoGen:
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 19


# ## Defining image specifications
#
# Here, we define two image specifications for the workflow:
# 1. The first image installs CUDA and is used for video generation. We're using a pre-release version of Diffusers since Mochi is available in this version.
# 2. The second image is used to download the model and run the dynamic workflow that processes the prompts.

image = fl.ImageSpec(
    name="genmo",
    packages=[
        "torch==2.5.1",
        "git+https://github.com/huggingface/diffusers.git@805aa93789fe9c95dd8d5a3ceac100d33f584ec7",
        "git+https://github.com/flyteorg/flytekit.git@650efe4425c799eaf66384575cc0e67521e9a851",  # PR: https://github.com/flyteorg/flytekit/pull/2931
        "transformers==4.46.3",
        "accelerate==1.1.1",
        "sentencepiece==0.2.0",
        "opencv-python==4.10.0.84",
    ],
    conda_channels=["nvidia"],
    conda_packages=[
        "cuda=12.1.0",
        "cuda-nvcc",
        "cuda-version=12.1.0",
        "cuda-command-line-tools=12.1.0",
    ],
    apt_packages=["git", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"],
)

image_with_no_cuda = fl.ImageSpec(
    name="genmo-no-cuda",
    packages=[
        "huggingface-hub==0.26.2",
        "git+https://github.com/flyteorg/flytekit.git@650efe4425c799eaf66384575cc0e67521e9a851",  # PR: https://github.com/flyteorg/flytekit/pull/2931
        "diffusers==0.31.0",
    ],
    apt_packages=["git"],
)

# ## Defining an actor environment
#
# The actor environment is used to retain the downloaded model across all actor executions.
# We set the accelerator to `A100` and the replica count to 1 to avoid downloading the model multiple times.

actor = ActorEnvironment(
    name="genmo-video-generation",
    replica_count=1,
    ttl_seconds=900,
    requests=fl.Resources(gpu="1", mem="100Gi"),
    container_image=image,
    accelerator=A100,
)

# ## Downloading the model
#
# The download step ensures that the model is cached and doesn't need to be downloaded from the HuggingFace hub
# every time this execution runs.


@fl.task(
    cache=True,
    cache_version="0.1",
    requests=fl.Resources(cpu="5", mem="45Gi"),  
    container_image=image_with_no_cuda,
)
def download_model(repo_id: str) -> FlyteDirectory:
    from huggingface_hub import snapshot_download

    ctx = fl.current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    snapshot_download(repo_id=repo_id, local_dir=cached_model_dir)
    return FlyteDirectory(path=cached_model_dir)


# ## Defining an actor task
#
# We define an actor task to generate a video using the Mochi 1 model.
# The model is downloaded once to a hard-coded path and used for every prompt.
# In the future, we plan to allow avoiding model initialization and loading onto a GPU every time.
# `enable_model_cpu_offload` offloads the model to CPU using accelerate, reducing memory usage with minimal performance impact.
# `enable_vae_tiling` saves a large amount of memory and allows processing larger images.


@actor.task
def genmo_video_generation(model_dir: FlyteDirectory, param_set: VideoGen) -> FlyteFile:
    import torch
    from diffusers import MochiPipeline
    from diffusers.utils import export_to_video

    local_path = Path("/tmp/genmo_mochi_model")

    if not local_path.exists():
        print("Model doesn't exist")
        ctx = FlyteContextManager.current_context()
        ctx.file_access.get_data(
            remote_path=model_dir.remote_source,
            local_path=local_path,
            is_multipart=True,
        )

    pipe = MochiPipeline.from_pretrained(
        local_path, variant="bf16", torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()

    frames = pipe(
        param_set.prompt,
        negative_prompt=param_set.negative_prompt,
        num_frames=param_set.num_frames,
    ).frames[0]

    ctx = fl.current_context()
    working_dir = Path(ctx.working_directory)
    video_file = working_dir / "video.mp4"
    export_to_video(frames, video_file, fps=30)

    return FlyteFile(path=video_file)


# ## Defining a dynamic workflow
#
# We define a dynamic workflow to loop through the prompts and parameters.
# It calls the actor task to generate the video.


@fl.dynamic(container_image=image_with_no_cuda)
def generate_videos(
    model_dir: FlyteDirectory, video_gen_params: list[VideoGen]
) -> list[FlyteFile]:
    videos = []
    for param_set in video_gen_params:
        videos.append(genmo_video_generation(model_dir=model_dir, param_set=param_set))
    return videos


# ## Defining a workflow
#
# With all tasks in place, we define a workflow to generate videos.
# Initialize `VideoGen` objects to specify the prompt, number of frames, and a negative prompt.


@fl.workflow
def genmo_video_generation_with_actor(
    repo_id: str = "genmo/mochi-1-preview",
    video_gen_params: list[VideoGen] = [
        VideoGen(
            prompt="A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl filled with lemons and sprigs of mint against a peach-colored background. The hand gently tosses the lemon up and catches it, showcasing its smooth texture. A beige string bag sits beside the bowl, adding a rustic touch to the scene. Additional lemons, one halved, are scattered around the base of the bowl. The even lighting enhances the vibrant colors and creates a fresh, inviting atmosphere.",
        ),
        VideoGen(
            prompt="Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.",
            num_frames=84,
        ),
    ],
) -> list[FlyteFile]:
    model_dir = download_model(repo_id=repo_id)
    return generate_videos(model_dir=model_dir, video_gen_params=video_gen_params)
