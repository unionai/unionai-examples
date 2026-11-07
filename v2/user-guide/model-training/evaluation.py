# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "scikit-learn",
#    "pandas",
#    "pyarrow",
#    "joblib",
# ]
# main = "evaluate_model"
# params = ""
# ///
"""Model evaluation: a sharded inference pass over a held-out set, metrics computed
from the stored predictions, and a gate that compares the candidate with a baseline."""

# {{docs-fragment env}}
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import flyte
import joblib
import pandas as pd
from flyte.io import File

image = flyte.Image.from_uv_script(__file__, name="evaluation")

# In a real pipeline, give this environment the GPU the model needs.
predict_env = flyte.TaskEnvironment(
    name="eval-predict",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

metrics_env = flyte.TaskEnvironment(
    name="eval-metrics",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

driver_env = flyte.TaskEnvironment(
    name="eval-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[predict_env, metrics_env],
)
# {{/docs-fragment env}}


@predict_env.task(cache="auto")
async def make_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data once into a training set and a held-out test set."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    df = load_breast_cancer(as_frame=True).frame
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["target"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


@predict_env.task(cache="auto")
async def train_model(train: pd.DataFrame, max_depth: int) -> File:
    """Stand-in for a training pipeline: fit a model and save it as a file."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=0)
    model.fit(train.drop(columns="target"), train["target"])
    path = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(model, path)
    return await File.from_local(str(path))


# {{docs-fragment predict}}
@predict_env.task(cache="auto", retries=2)
async def predict(model: File, shard: pd.DataFrame) -> pd.DataFrame:
    """Run the model over one shard of the held-out set and keep every prediction."""
    clf = joblib.load(await model.download())
    features = shard.drop(columns="target")
    return pd.DataFrame(
        {
            "label": shard["target"],
            "prediction": clf.predict(features),
            "score": clf.predict_proba(features)[:, 1],
        }
    )
# {{/docs-fragment predict}}


# {{docs-fragment metrics}}
@dataclass
class Metrics:
    accuracy: float
    f1: float
    roc_auc: float


@metrics_env.task
async def compute_metrics(predictions: pd.DataFrame) -> Metrics:
    """Compute metrics from stored predictions. Changing a metric reruns only this task."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y, y_hat = predictions["label"], predictions["prediction"]
    return Metrics(
        accuracy=float(accuracy_score(y, y_hat)),
        f1=float(f1_score(y, y_hat)),
        roc_auc=float(roc_auc_score(y, predictions["score"])),
    )
# {{/docs-fragment metrics}}


# {{docs-fragment evaluate}}
async def run_eval(model: File, test: pd.DataFrame, n_shards: int) -> Metrics:
    shards = [test.iloc[i::n_shards].reset_index(drop=True) for i in range(n_shards)]
    predictions = await asyncio.gather(*(predict(model, s) for s in shards))
    return await compute_metrics(pd.concat(predictions, ignore_index=True))


@driver_env.task(report=True)
async def evaluate_model(max_depth: int = 8, baseline_depth: int = 2, n_shards: int = 4) -> bool:
    train, test = await make_split()
    candidate, baseline = await asyncio.gather(train_model(train, max_depth), train_model(train, baseline_depth))

    with flyte.group("candidate"):
        new = await run_eval(candidate, test, n_shards)
    with flyte.group("baseline"):
        old = await run_eval(baseline, test, n_shards)

    rows = "".join(
        f"<tr><td>{name}</td><td>{getattr(old, name):.4f}</td><td>{getattr(new, name):.4f}</td></tr>"
        for name in ("accuracy", "f1", "roc_auc")
    )
    await flyte.report.replace.aio(
        f"<h2>Evaluation</h2><table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr>{rows}</table>",
        do_flush=True,
    )

    # Promote the candidate only if it beats the baseline on the metric that matters.
    return new.roc_auc >= old.roc_auc
# {{/docs-fragment evaluate}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(evaluate_model)
    print(run.url)
