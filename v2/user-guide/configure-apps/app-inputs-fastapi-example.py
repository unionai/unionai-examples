"""Example: FastAPI app with configurable model input."""

from contextlib import asynccontextmanager
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI
import os
import flyte
import joblib


# {{docs-fragment model-serving-api}}
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Access input via environment variable
    model = joblib.load(os.getenv("MODEL_PATH", "/app/models/default.pkl"))
    state["model"] = model
    yield


app = FastAPI(lifespan=lifespan)

app_env = FastAPIAppEnvironment(
    name="model-serving-api",
    app=app,
    inputs=[
        flyte.app.Input(
            name="model_file",
            value=flyte.io.File("s3://bucket/models/default.pkl"),
            mount="/app/models",
            env_var="MODEL_PATH",
        ),
    ],
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn", "scikit-learn"
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=False,
)

@app.get("/predict")
async def predict(data: dict):
    model = state["model"]
    return {"prediction": model.predict(data)}
# {{/docs-fragment model-serving-api}}

