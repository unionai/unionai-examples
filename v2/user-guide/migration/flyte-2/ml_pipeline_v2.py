# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "pandas",
#    "scikit-learn",
#    "joblib",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import os

import joblib
import pandas as pd
import flyte
from flyte.io import File
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Image, resources, and cache are set once on the TaskEnvironment.
env = flyte.TaskEnvironment(
    name="ml_pipeline",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas", "scikit-learn", "joblib"
    ),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    cache="auto",
)


@env.task
async def load_data() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame
    df["species"] = data.target
    return df


@env.task
async def train_model(data: pd.DataFrame) -> File:
    model = RandomForestClassifier()
    X = data.drop("species", axis=1)
    y = data["species"]
    model.fit(X, y)

    model_path = os.path.join(os.getcwd(), "model.joblib")
    joblib.dump(model, model_path)
    return await File.from_local(model_path)


@env.task
async def evaluate(model_file: File, data: pd.DataFrame) -> float:
    local_path = await model_file.download()
    model = joblib.load(local_path)
    X = data.drop("species", axis=1)
    y = data["species"]
    return float(model.score(X, y))


# The "workflow" is just an orchestrating task.
@env.task
async def main() -> float:
    data = await load_data()
    model = await train_model(data)
    return await evaluate(model, data)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
