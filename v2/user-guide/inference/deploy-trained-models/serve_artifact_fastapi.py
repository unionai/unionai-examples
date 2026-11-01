# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "fastapi",
#    "uvicorn",
#    "joblib",
#    "scikit-learn",
# ]
# ///

"""Serve a classifier artifact with FastAPI."""

# {{docs-fragment app}}
from contextlib import asynccontextmanager

import flyte
import flyte.app
import joblib
from fastapi import FastAPI
from flyte.app import Parameter
from flyte.app.extras import FastAPIAppEnvironment
from flyte.remote import App

MODEL_DIR = "/tmp/model"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(f"{MODEL_DIR}/model.joblib")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(features: list[float]) -> dict:
    return {"prediction": state["model"].predict([features]).tolist()[0]}


app_env = FastAPIAppEnvironment(
    name="churn-classifier",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "joblib", "scikit-learn"),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.ArtifactValue(name="churn-classifier", type="directory"),
            mount=MODEL_DIR,
        ),
    ],
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(app_env)
    App.get(name=app_env.name).activate()
# {{/docs-fragment app}}
