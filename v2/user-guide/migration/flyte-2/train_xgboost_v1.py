import os

import joblib
from flytekit import task, workflow, ImageSpec, Resources, current_context
from flytekit.types.file import FlyteFile
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

image = ImageSpec(
    name="xgb-image",
    packages=["xgboost", "scikit-learn", "joblib"],
)


@task(container_image=image, requests=Resources(cpu="2", mem="4Gi"))
def train_model(n_estimators: int, max_depth: int) -> FlyteFile:
    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, random_state=42)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    model_path = os.path.join(current_context().working_directory, "model.json")
    joblib.dump(model, model_path)
    return FlyteFile(path=model_path)


@task(container_image=image)
def evaluate(model_file: FlyteFile) -> float:
    model = joblib.load(model_file.download())
    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(data.data, data.target, random_state=42)
    return float(model.score(X_test, y_test))


@workflow
def main(n_estimators: int, max_depth: int) -> float:
    model = train_model(n_estimators=n_estimators, max_depth=max_depth)
    return evaluate(model_file=model)
