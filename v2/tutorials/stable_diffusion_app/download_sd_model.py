"""
Download Stable Diffusion model and upload to S3.

This workflow downloads the Stable Diffusion model from Hugging Face
and saves it to object storage (S3) for use by the image generator app.
"""
import asyncio
from pathlib import Path

import flyte
import flyte.io


# Create image with model dependencies
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
)

env = flyte.TaskEnvironment(
    name="download_sd_model",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", disk="20Gi"),
)


@env.task()
async def download_stable_diffusion_model(
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> flyte.io.Dir:
    """
    Download Stable Diffusion model from Hugging Face and save to S3.

    Args:
        model_id: Hugging Face model ID to download

    Returns:
        Directory containing the downloaded model (automatically uploaded to S3)
    """
    print(f"Downloading Stable Diffusion model: {model_id}")

    import torch
    from diffusers import StableDiffusionPipeline

    # Download model from Hugging Face
    # This will cache it locally first
    print("Fetching model from Hugging Face...")
    pipe = await asyncio.to_thread(
        StableDiffusionPipeline.from_pretrained,
        model_id,
        torch_dtype=torch.float16,
    )

    # Save to local directory
    output_dir = Path("/tmp/stable_diffusion_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {output_dir}")
    await asyncio.to_thread(pipe.save_pretrained, str(output_dir))

    print("Model saved successfully. Uploading to S3...")

    # Upload to S3 and return reference
    # This automatically uploads to object storage and returns a remote reference
    model_dir = await flyte.io.Dir.from_local(output_dir)

    print(f"Model uploaded to S3: {model_dir.path}")
    return model_dir


if __name__ == "__main__":
    flyte.init_from_config()

    # Run the download workflow
    print("Starting model download workflow...")
    run = flyte.run(download_stable_diffusion_model)
    print(f"Run Name: {run.name}")
    print(f"Run URL: {run.url}")
