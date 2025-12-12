"""Example: FastAPI app with configurable model input."""

from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI
import os
import flyte

app = FastAPI()

# {{docs-fragment model-serving-api}}
# Access input via environment variable
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/default.pkl")

app_env = FastAPIAppEnvironment(
    name="model-serving-api",
    app=app,
    inputs=[
        flyte.app.Input(
            name="model_file",
            value=flyte.File("s3://bucket/models/default.pkl"),
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
    # Load model from MODEL_PATH
    # model = load_model(MODEL_PATH)
    # return model.predict(data)
    return {"prediction": "example"}
# {{/docs-fragment model-serving-api}}

