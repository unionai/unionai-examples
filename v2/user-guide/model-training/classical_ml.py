# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "scikit-learn",
#    "pandas",
#    "pyarrow",
#    "joblib",
# ]
# main = "training_pipeline"
# params = ""
# ///
"""A classical ML pipeline: cached data prep, a parallel hyperparameter sweep, and a final fit."""

# {{docs-fragment env}}
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import flyte
import joblib
import pandas as pd
from flyte.io import File

# The image installs the dependencies listed in this file's script header.
image = flyte.Image.from_uv_script(__file__, name="classical-ml")

env = flyte.TaskEnvironment(
    name="classical-ml",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

driver_env = flyte.TaskEnvironment(
    name="classical-ml-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[env],
)
# {{/docs-fragment env}}


# {{docs-fragment data}}
@env.task(cache="auto")
async def load_data() -> pd.DataFrame:
    """Load and clean the training data once. Later runs reuse the cached output."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    return data.frame.dropna()
# {{/docs-fragment data}}


# {{docs-fragment trial}}
@dataclass
class Params:
    n_estimators: int
    max_depth: int


@dataclass
class TrialResult:
    params: Params
    score: float


@env.task(cache="auto", retries=2)
async def evaluate(df: pd.DataFrame, params: Params) -> TrialResult:
    """Score one hyperparameter setting with cross-validation."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X, y = df.drop(columns="target"), df["target"]
    model = GradientBoostingClassifier(
        n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=0
    )
    score = cross_val_score(model, X, y, cv=5).mean()
    return TrialResult(params=params, score=float(score))
# {{/docs-fragment trial}}


# {{docs-fragment fit}}
@env.task
async def fit(df: pd.DataFrame, params: Params) -> File:
    """Fit the final model on all of the data and save it as a file."""
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = df.drop(columns="target"), df["target"]
    model = GradientBoostingClassifier(
        n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=0
    ).fit(X, y)
    path = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(model, path)
    return await File.from_local(str(path))
# {{/docs-fragment fit}}


# {{docs-fragment pipeline}}
@driver_env.task(report=True)
async def training_pipeline(
    n_estimators: list[int] = [50, 100, 200],
    max_depth: list[int] = [2, 3, 4],
) -> File:
    df = await load_data()

    # One task per setting, all running at once.
    grid = [Params(n, d) for n in n_estimators for d in max_depth]
    with flyte.group("sweep"):
        results = await asyncio.gather(*(evaluate(df, p) for p in grid))

    best = max(results, key=lambda r: r.score)
    rows = "".join(
        f"<tr><td>{r.params.n_estimators}</td><td>{r.params.max_depth}</td><td>{r.score:.4f}</td></tr>"
        for r in sorted(results, key=lambda r: -r.score)
    )
    await flyte.report.replace.aio(
        f"<h2>Sweep results</h2><table><tr><th>n_estimators</th><th>max_depth</th><th>CV accuracy</th></tr>"
        f"{rows}</table>",
        do_flush=True,
    )
    return await fit(df, best.params)
# {{/docs-fragment pipeline}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(training_pipeline)
    print(run.url)
