# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
#    "scikit-learn",
#    "joblib",
# ]
# ///

"""Example of serving a machine learning model with FastAPI."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import flyte
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel


# {{docs-fragment ml-model}}
app = FastAPI(title="ML Model API")


# Define request/response models
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float


class PredictionResponse(BaseModel):
    prediction: float
    probability: float


# Load model (you would typically load this from storage)
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/model.joblib")
    # In production, load from your storage
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    yield


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Make prediction
    # prediction = model.predict([[request.feature1, request.feature2, request.feature3]])

    # Dummy prediction for demo
    prediction = 0.85
    probability = 0.92

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
    )


env = FastAPIAppEnvironment(
    name="ml-model-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "scikit-learn",
        "pydantic",
        "joblib",
    ),
    parameters=[
        flyte.app.Parameter(
            name="model_file",
            value=flyte.io.File("s3://bucket/models/model.joblib"),
            mount="/app/models",
            env_var="MODEL_PATH",
        ),
    ],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=False,
)
# {{/docs-fragment ml-model}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"API URL: {app_deployment[0].url}")
    print(f"Swagger docs: {app_deployment[0].url}/docs")
