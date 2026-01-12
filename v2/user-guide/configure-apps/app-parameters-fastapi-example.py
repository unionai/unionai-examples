# /// script
# requires-python = "==3.13"
# dependencies = [
#    "fastapi",
#    "uvicorn",
#    "joblib",
#    "scikit-learn",
#    "flyte==2.0.0b45",
# ]
# ///

from contextlib import asynccontextmanager
from pathlib import Path

import flyte
import flyte.app
import flyte.io
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI


# {{docs-fragment model-serving-api}}

image = flyte.Image.from_uv_script(__file__, name="app-parameters-fastapi-example")

task_env = flyte.TaskEnvironment(
    name="model_serving_task",
    image=image,
    resources=flyte.Resources(cpu=2, memory="1Gi"),
    cache="auto",
)

@task_env.task
async def train_model_task() -> flyte.io.File:
    """Train a model and return it."""
    import joblib
    import sklearn.ensemble
    import sklearn.datasets

    X, y = sklearn.datasets.make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)

    model_dir = Path("/tmp/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    return await flyte.io.File.from_local(model_path)


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import joblib

    model = joblib.load("/root/models/model.joblib")
    state["model"] = model
    yield


app = FastAPI(lifespan=lifespan)

app_env = FastAPIAppEnvironment(
    name="model-serving-api",
    app=app,
    parameters=[
        flyte.app.Parameter(
            name="model_file",
            # this is a placeholder
            value=flyte.io.File.from_existing_remote("s3://bucket/models/default.pkl"),
            mount="/root/models/",
            download=True,
        ),
    ],
    image=image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    requires_auth=False,
)

@app.post("/predict")
async def predict(data: list[float]) -> dict[str, list[float]]:
    model = state["model"]
    return {"prediction": model.predict([data]).tolist()}


if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)

    run = flyte.run(train_model_task)
    print(f"Run: {run.url}")
    run.wait()

    model_file = run.outputs()[0]
    print(f"Model file: {model_file.path}")

    app = flyte.with_servecontext(
        parameter_values={
            "model-serving-api": {
                "model_file": flyte.io.File.from_existing_remote(model_file.path)
            }
        }
    ).serve(app_env)
    print(f"API URL: {app.url}")
# {{/docs-fragment model-serving-api}}
