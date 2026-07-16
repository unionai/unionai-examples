# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "scikit-learn",
#    "joblib",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import asyncio
import os

import joblib
import flyte
from flyte.io import File
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

env = flyte.TaskEnvironment(
    name="batch_inference",
    image=flyte.Image.from_debian_base().with_pip_packages("scikit-learn", "joblib"),
)


@env.task
async def train_model() -> File:
    data = load_iris()
    model = RandomForestClassifier().fit(data.data, data.target)
    model_path = os.path.join(os.getcwd(), "model.joblib")
    joblib.dump(model, model_path)
    return await File.from_local(model_path)


@env.task
async def score_batch(model_file: File, batch: list[list[float]]) -> list[int]:
    local_path = await model_file.download()
    model = joblib.load(local_path)
    return [int(p) for p in model.predict(batch)]


@env.task
async def main() -> list[list[int]]:
    model = await train_model()
    rows = load_iris().data.tolist()
    batches = [rows[i : i + 30] for i in range(0, len(rows), 30)]
    # Score every batch in parallel, reusing the same model reference.
    coros = [score_batch(model, batch) for batch in batches]
    return list(await asyncio.gather(*coros))
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
