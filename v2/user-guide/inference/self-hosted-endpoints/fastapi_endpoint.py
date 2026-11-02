# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "fastapi",
#    "uvicorn",
#    "torch",
#    "transformers",
# ]
# ///

"""Serve a Hugging Face model behind a custom FastAPI endpoint."""

from pathlib import Path

# {{docs-fragment api}}
from contextlib import asynccontextmanager

import flyte
import flyte.app
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel
from transformers import pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once per replica, before the app accepts traffic.
    state["classifier"] = pipeline("text-classification", model=MODEL)
    yield


app = FastAPI(title="Sentiment endpoint", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(**state["classifier"](request.text)[0])
# {{/docs-fragment api}}


# {{docs-fragment app-env}}
app_env = FastAPIAppEnvironment(
    name="fastapi-endpoint",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn", "torch", "transformers"
    ),
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 2), scaledown_after=300),
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Deployed: {deployment.url}")
    print(f"API docs: {deployment.url}/docs")
# {{/docs-fragment deploy}}
