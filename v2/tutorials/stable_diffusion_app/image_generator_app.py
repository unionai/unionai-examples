"""
GPU-powered image generation app using Stable Diffusion.

This Flyte app exposes an HTTP endpoint that generates images from text prompts
using open-source diffusion models.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import flyte
import flyte.io
from flyte.app import Input, Scaling
from flyte.app.extras import FastAPIAppEnvironment

MODEL_DIR_ENV = "MODEL_DIR"

# Create image with all dependencies
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "fastapi",
    "uvicorn",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    """
    # Startup: Load model
    print("Starting up: Loading Stable Diffusion model...")

    # Get model path from environment inputs
    # In production, this would come from the deployed app's mounted inputs
    # For local testing, you can set a path here
    model_path = Path("/tmp/stable_diffusion_model")  # Default for local testing

    if model_path.exists():
        await load_model(model_path)
        print("Startup complete: Model loaded and ready")
    else:
        print(f"Model not found at {model_path}. App will start but won't be ready.")

    yield

    # Shutdown: Clean up resources if needed
    print("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Image Generator",
    description="Generate images from text prompts using Stable Diffusion",
    version="1.0.0",
    lifespan=lifespan,
)

# Application state - initialized here to provide type hints
app.state.pipeline: Optional[object] = None  # Will hold StableDiffusionPipeline


async def load_model(model_dir: Path):
    """
    Load the Stable Diffusion model from a directory.

    Args:
        model_dir: Path to directory containing the pre-downloaded model
    """
    import torch
    from diffusers import StableDiffusionPipeline

    print(f"Loading Stable Diffusion model from {model_dir}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model from local directory in a background thread
    pipe = await asyncio.to_thread(
        StableDiffusionPipeline.from_pretrained,
        str(model_dir),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()

    # Store in app state
    app.state.pipeline = pipe

    print(f"Model loaded successfully on {device}!")

# Create Flyte app environment with GPU support
env = FastAPIAppEnvironment(
    name="image-generator",
    app=app,
    description="Stable Diffusion image generation service",
    image=image,
    resources=flyte.Resources(gpu=1, memory="16Gi", disk="5Gi"),
    requires_auth=False,
    scaling=Scaling(replicas=(1, 1)),  # Keep exactly 1 replica running to avoid cold starts
    inputs=[
        Input(
            name="model",
            value=flyte.io.Dir.from_existing_remote(
                os.environ.get(MODEL_DIR_ENV, "/tmp/stable_diffusion_model")
            ),
            mount="/tmp/stable_diffusion_model",
        )
    ],
)


class ImageRequest(BaseModel):
    """Request model for image generation."""
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512


@app.post("/generate")
async def generate_image(image_request: ImageRequest):
    """
    Generate an image from a text prompt using Stable Diffusion.

    Args:
        image_request: ImageRequest with prompt and generation parameters

    Returns:
        PNG image file
    """
    if app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    print(f"Generating image for prompt: {image_request.prompt}")

    # Generate image in background thread (GPU operations are blocking)
    image = await asyncio.to_thread(
        app.state.pipeline,
        prompt=image_request.prompt,
        negative_prompt=image_request.negative_prompt if image_request.negative_prompt else None,
        num_inference_steps=image_request.num_inference_steps,
        guidance_scale=image_request.guidance_scale,
        width=image_request.width,
        height=image_request.height,
    )
    image = image.images[0]

    # Save to temporary file
    output_path = "/tmp/generated_image.png"
    await asyncio.to_thread(image.save, output_path)
    print(f"Image saved to {output_path}")

    # Return as FastAPI FileResponse
    return FileResponse(
        output_path,
        media_type="image/png",
        filename="generated_image.png",
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch

    return {
        "status": "healthy" if app.state.pipeline is not None else "not_ready",
        "service": "image-generator",
        "model_loaded": app.state.pipeline is not None,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)

    # Deploy the FastAPI app to Flyte
    # Note: You need to provide a pre-downloaded model directory
    # You can download the model using a separate workflow and get its S3 path:
    #   run = flyte.run(download_model_workflow)
    #   model_dir = run.outputs()
    #   then set `export MODEL_DIR="s3://...`
    # then run this script
    v = os.environ.get(MODEL_DIR_ENV, None)
    if v is None:
        raise RuntimeError(
            f"Model directory not found: {v}, please set {MODEL_DIR_ENV} as an environment variable"
        )
    app_deployment = flyte.serve(env)
    print(f"Deployed Application: {app_deployment.url}")
