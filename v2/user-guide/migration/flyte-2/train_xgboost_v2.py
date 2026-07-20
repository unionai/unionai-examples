# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "xgboost",
#    "scikit-learn",
#    "joblib",
# ]
# main = "main"
# params = "n_estimators=50 max_depth=3"
# ///

# {{docs-fragment all}}
import os

import joblib
import flyte
from flyte.io import File
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

env = flyte.TaskEnvironment(
    name="train_xgboost",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "xgboost", "scikit-learn", "joblib"
    ),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)


@env.task
async def train_model(n_estimators: int, max_depth: int) -> File:
    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, random_state=42)
    model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    model_path = os.path.join(os.getcwd(), "model.json")
    joblib.dump(model, model_path)
    return await File.from_local(model_path)


@env.task
async def evaluate(model_file: File) -> float:
    local_path = await model_file.download()
    model = joblib.load(local_path)
    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(data.data, data.target, random_state=42)
    return float(model.score(X_test, y_test))


@env.task
async def main(n_estimators: int, max_depth: int) -> float:
    model = await train_model(n_estimators, max_depth)
    return await evaluate(model)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, n_estimators=50, max_depth=3)
    print(r.name)
    print(r.url)
    r.wait()
