import os
from functools import partial

import joblib
from flytekit import task, workflow, map_task, ImageSpec, current_context
from flytekit.types.file import FlyteFile
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

image = ImageSpec(name="inference-image", packages=["scikit-learn", "joblib"])


@task(container_image=image)
def train_model() -> FlyteFile:
    data = load_iris()
    model = RandomForestClassifier().fit(data.data, data.target)
    model_path = os.path.join(current_context().working_directory, "model.joblib")
    joblib.dump(model, model_path)
    return FlyteFile(path=model_path)


@task(container_image=image)
def get_batches() -> list[list[list[float]]]:
    data = load_iris()
    rows = data.data.tolist()
    # Split the rows into batches of 30.
    return [rows[i : i + 30] for i in range(0, len(rows), 30)]


@task(container_image=image)
def score_batch(model_file: FlyteFile, batch: list[list[float]]) -> list[int]:
    model = joblib.load(model_file.download())
    return [int(p) for p in model.predict(batch)]


@workflow
def main() -> list[list[int]]:
    model = train_model()
    batches = get_batches()
    return map_task(partial(score_batch, model_file=model))(batch=batches)
