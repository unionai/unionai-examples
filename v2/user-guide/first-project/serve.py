# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte==2.0.0b44",
#    "fastapi>=0.104.0",
#    "uvicorn>=0.24.0",
#    "torch>=2.0.0",
#    "transformers>=4.35.0",
#    "pydantic>=2.0.0",
# ]
# ///

"""
Model Serving API

Serves the fine-tuned DistilGPT-2 model via a FastAPI endpoint for text generation.
The model is loaded from the training pipeline output.
"""

# {{docs-fragment serve-imports}}
import os
import tarfile
import tempfile

import flyte
from fastapi import FastAPI, HTTPException
from flyte.app import Parameter, RunOutput
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel, Field

# {{/docs-fragment serve-imports}}


# {{docs-fragment serve-models}}
class GenerationRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(..., description="The input text to continue generating from")
    max_length: int = Field(
        default=100, ge=10, le=500, description="Maximum length of generated text"
    )
    temperature: float = Field(
        default=0.7, ge=0.1, le=2.0, description="Sampling temperature (higher = more random)"
    )
    top_k: int = Field(
        default=50, ge=1, le=100, description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=0.95, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter"
    )


class GenerationResponse(BaseModel):
    """Response model for text generation."""

    prompt: str = Field(..., description="The original input prompt")
    generated_text: str = Field(..., description="The generated text continuation")
    model_name: str = Field(..., description="Name of the model used")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool


# {{/docs-fragment serve-models}}

# {{docs-fragment serve-app}}
app = FastAPI(
    title="Text Generation API",
    description="Generate text continuations using a fine-tuned DistilGPT-2 model",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if the service is healthy and model is loaded."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text continuation from the given prompt."""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = app.state.model
    tokenizer = app.state.tokenizer

    # Tokenize input
    inputs = tokenizer(request.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    outputs = model.generate(
        **inputs,
        max_length=request.max_length,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return GenerationResponse(
        prompt=request.prompt,
        generated_text=generated_text,
        model_name="distilgpt2-finetuned",
    )


# {{/docs-fragment serve-app}}

# {{docs-fragment serve-env}}
env = FastAPIAppEnvironment(
    name="text-generation-api",
    app=app,
    description="Text generation API serving a fine-tuned DistilGPT-2 model",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pydantic>=2.0.0",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    parameters=[
        Parameter(
            name="model",
            value=RunOutput(task_name="model-training.training_pipeline", type="file"),
            download=True,
            env_var="MODEL_PATH",
        ),
    ],
    requires_auth=False,
)
# {{/docs-fragment serve-env}}


# {{docs-fragment serve-server}}
@env.server
async def init_server():
    """
    Initialize the server by loading the trained model.
    This runs once when the server starts up.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_archive_path = os.environ.get("MODEL_PATH")
    if not model_archive_path:
        print("Warning: MODEL_PATH not set, model will not be loaded")
        return

    print(f"Loading model from {model_archive_path}...")

    # Extract the model archive
    extract_dir = tempfile.mkdtemp()
    with tarfile.open(model_archive_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    model_path = os.path.join(extract_dir, "model")
    print(f"Model extracted to {model_path}")

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Store in app state for use by endpoints
    app.state.model = model
    app.state.tokenizer = tokenizer

    print("Model loaded successfully!")


# {{/docs-fragment serve-server}}

# {{docs-fragment serve-main}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy the serving app
    # The model parameter will be resolved from the most recent training_pipeline run
    print("Deploying text generation API...")
    deployment = flyte.serve(env)
    print(f"API deployed at: {deployment.url}")
    print(f"Swagger docs: {deployment.url}/docs")
# {{/docs-fragment serve-main}}
