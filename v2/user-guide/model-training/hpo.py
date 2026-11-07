# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "scikit-learn",
#    "pandas",
#    "pyarrow",
#    "joblib",
# ]
# main = "successive_halving"
# params = ""
# ///
"""Hyperparameter optimization: a random search that runs every trial as a task,
and successive halving, which continues only the best trials from their checkpoints."""

# {{docs-fragment env}}
import asyncio
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import flyte
import joblib
import pandas as pd
from flyte.io import File

image = flyte.Image.from_uv_script(__file__, name="hpo")

trial_env = flyte.TaskEnvironment(
    name="hpo-trial",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

driver_env = flyte.TaskEnvironment(
    name="hpo-driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[trial_env],
)
# {{/docs-fragment env}}


@trial_env.task(cache="auto")
async def load_data() -> pd.DataFrame:
    from sklearn.datasets import load_digits

    data = load_digits(as_frame=True)
    return data.frame


# {{docs-fragment trial}}
@dataclass
class Params:
    alpha: float
    eta0: float


@dataclass
class Trial:
    params: Params
    epochs: int
    score: float
    checkpoint: File


@trial_env.task(cache="auto", retries=2)
async def train_trial(
    df: pd.DataFrame, params: Params, epochs: int, start_epoch: int = 0, checkpoint: File | None = None
) -> Trial:
    """Train one setting up to `epochs`, continuing from `checkpoint` if one is given."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split

    X, y = df.drop(columns="target").to_numpy() / 16.0, df["target"].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)

    if checkpoint is not None:
        model = joblib.load(await checkpoint.download())
    else:
        model = SGDClassifier(alpha=params.alpha, eta0=params.eta0, learning_rate="constant", random_state=0)

    for _ in range(start_epoch, epochs):
        model.partial_fit(X_train, y_train, classes=sorted(set(y)))

    path = Path(tempfile.mkdtemp()) / "model.joblib"
    joblib.dump(model, path)
    return Trial(
        params=params,
        epochs=epochs,
        score=float(model.score(X_val, y_val)),
        checkpoint=await File.from_local(str(path)),
    )
# {{/docs-fragment trial}}


# {{docs-fragment random-search}}
def sample_params(rng: random.Random) -> Params:
    return Params(alpha=10 ** rng.uniform(-6, -2), eta0=10 ** rng.uniform(-3, 0))


@driver_env.task
async def random_search(n_trials: int = 16, max_parallel: int = 8, seed: int = 0) -> Params:
    df = await load_data()
    rng = random.Random(seed)
    limit = asyncio.Semaphore(max_parallel)

    async def run_one(i: int, params: Params) -> Trial:
        async with limit:
            with flyte.group(f"trial-{i}"):
                return await train_trial(df, params, epochs=20)

    trials = await asyncio.gather(*(run_one(i, sample_params(rng)) for i in range(n_trials)))
    return max(trials, key=lambda t: t.score).params
# {{/docs-fragment random-search}}


# {{docs-fragment halving}}
@driver_env.task(report=True)
async def successive_halving(n_trials: int = 16, min_epochs: int = 2, eta: int = 2, seed: int = 0) -> Trial:
    df = await load_data()
    rng = random.Random(seed)
    trials = [None] * n_trials
    params = [sample_params(rng) for _ in range(n_trials)]
    rounds = int(math.log(n_trials, eta))
    history = []

    epochs = min_epochs
    for rung in range(rounds + 1):
        with flyte.group(f"rung-{rung}"):
            trials = await asyncio.gather(
                *(
                    train_trial(
                        df,
                        p,
                        epochs=epochs,
                        start_epoch=t.epochs if t else 0,
                        checkpoint=t.checkpoint if t else None,
                    )
                    for p, t in zip(params, trials)
                )
            )
        history.append((rung, epochs, len(trials), max(t.score for t in trials)))
        if rung == rounds:
            break

        # Keep the best 1/eta of the trials and give each eta times the budget.
        survivors = sorted(trials, key=lambda t: -t.score)[: max(1, len(trials) // eta)]
        params = [t.params for t in survivors]
        trials = survivors
        epochs *= eta

    rows = "".join(f"<tr><td>{r}</td><td>{e}</td><td>{n}</td><td>{s:.4f}</td></tr>" for r, e, n, s in history)
    await flyte.report.replace.aio(
        "<h2>Successive halving</h2><table><tr><th>Rung</th><th>Epochs</th><th>Trials</th>"
        f"<th>Best accuracy</th></tr>{rows}</table>",
        do_flush=True,
    )
    return max(trials, key=lambda t: t.score)
# {{/docs-fragment halving}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(successive_halving)
    print(run.url)
