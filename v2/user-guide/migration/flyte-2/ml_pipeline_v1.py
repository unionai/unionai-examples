import os

import joblib
import pandas as pd
from flytekit import task, workflow, ImageSpec, Resources, current_context
from flytekit.types.file import FlyteFile
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

image = ImageSpec(
    name="ml-image",
    packages=["pandas", "scikit-learn", "joblib"],
)


@task(
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi"),
    cache=True,
    cache_version="1.0",
)
def load_data() -> pd.DataFrame:
    data = load_iris(as_frame=True)
    df = data.frame
    df["species"] = data.target
    return df


@task(container_image=image)
def train_model(data: pd.DataFrame) -> FlyteFile:
    model = RandomForestClassifier()
    X = data.drop("species", axis=1)
    y = data["species"]
    model.fit(X, y)

    model_path = os.path.join(current_context().working_directory, "model.joblib")
    joblib.dump(model, model_path)
    return FlyteFile(path=model_path)


@task(container_image=image)
def evaluate(model_file: FlyteFile, data: pd.DataFrame) -> float:
    model = joblib.load(model_file.download())
    X = data.drop("species", axis=1)
    y = data["species"]
    return float(model.score(X, y))


@workflow
def main() -> float:
    data = load_data()
    model = train_model(data=data)
    return evaluate(model_file=model, data=data)
