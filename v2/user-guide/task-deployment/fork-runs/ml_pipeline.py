# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.6.5",
#    "flyteplugins-union>=0.8.1",
# ]
# main = "main"
# params = ""
# ///

"""Fork a failed ML training pipeline: fix the code, keep the completed work.

Same pipeline as the recovery example — load, preprocess, train, then `evaluate` registers the
result with the validation service — but this time the failure is not a flaky dependency. It is
a bug in our own code: `evaluate` divides by the number of predictions that clear
`threshold`, and when a release evaluation runs with `threshold=1.0` (a legitimate "how many
predictions are *certain*?" query), nothing clears it and the task dies at the very last step
with a `ZeroDivisionError`.

Recovery cannot help here — it replays the source run's code as-is, so the recovered run would
hit the same division by zero. The fix is a code change, and `flyte fork` is the verb for it:
it replays the run with the code in your working tree, reusing every action whose code and
inputs are unchanged. Edit `evaluate`, fork the failed run, and load / preprocess / train are
reused while only `evaluate` re-executes.

This file ships in the buggy state (the code that was deployed when the run failed). The
corrected version lives in `ml_pipeline_fixed.py`, which also automates the whole tour: launch
the buggy pipeline, fork it with the fix, and print which actions were reused.

--------------------------------------------------------------------------------------------
The manual flow (what you would actually do)
--------------------------------------------------------------------------------------------
Requires `flyteplugins-union` (`pip install flyteplugins-union`), which ships `flyte fork`.

    # 1. Launch the pipeline as deployed. A threshold of 1.0 trips the bug in `evaluate`.
    flyte run ml_pipeline.py main --threshold 1.0
    # -> fails at evaluate: ZeroDivisionError: division by zero

    # 2. Fix `evaluate` in this file (the one-line change shown in ml_pipeline_fixed.py):
    #        confident_accuracy = confident_correct / confident_count if confident_count else 0.0

    # 3. Fork the failed run with your edited working tree. Actions whose code and inputs are
    #    unchanged — load_dataset, preprocess, train — are reused; only evaluate re-executes.
    flyte fork <RUN> ml_pipeline.py main --follow

    # Inspect what the fork reused — reused actions show up as RECOVERED:
    flyte get action <FORKED_RUN>

Or run the automated end-to-end tour:

    uv run python ml_pipeline_fixed.py
"""

import asyncio
import math
import random
from dataclasses import dataclass

import flyte
from flyte.models import ActionPhase
from flyte.remote import Action

env = flyte.TaskEnvironment(
    name="ml_pipeline",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)


@dataclass
class Dataset:
    features: list[list[float]]
    labels: list[int]


@dataclass
class Model:
    weights: list[float]
    bias: float


@dataclass
class EvaluationReport:
    accuracy: float
    confident_accuracy: float
    confident_count: int
    approved: bool


# {{docs-fragment pipeline}}
@env.task
async def load_dataset(n_samples: int, seed: int) -> Dataset:
    """Pull a synthetic training set from the (mock) feature store."""
    rng = random.Random(seed)
    features, labels = [], []
    for _ in range(n_samples):
        label = rng.randint(0, 1)
        center = 2.0 if label else -2.0
        features.append([rng.gauss(center, 1.0), rng.gauss(center, 1.0)])
        labels.append(label)
    print(f"load_dataset: loaded {n_samples} samples")
    await asyncio.sleep(2)  # stand-in for a slow warehouse read
    return Dataset(features=features, labels=labels)


@env.task
async def preprocess(data: Dataset) -> Dataset:
    """Standardize features to zero mean and unit variance."""
    n = len(data.features)
    means = [sum(f[j] for f in data.features) / n for j in range(2)]
    stds = [
        max(math.sqrt(sum((f[j] - means[j]) ** 2 for f in data.features) / n), 1e-9)
        for j in range(2)
    ]
    scaled = [[(f[j] - means[j]) / stds[j] for j in range(2)] for f in data.features]
    print(f"preprocess: standardized {n} rows")
    await asyncio.sleep(2)  # stand-in for an expensive feature transformation
    return Dataset(features=scaled, labels=data.labels)


@env.task
async def train(data: Dataset, epochs: int) -> Model:
    """Fit a small logistic-regression model with gradient descent."""
    w = [0.0, 0.0]
    b = 0.0
    lr = 0.1
    n = len(data.features)
    for _ in range(epochs):
        gw0 = gw1 = gb = 0.0
        for x, y in zip(data.features, data.labels):
            p = 1.0 / (1.0 + math.exp(-(w[0] * x[0] + w[1] * x[1] + b)))
            err = p - y
            gw0 += err * x[0]
            gw1 += err * x[1]
            gb += err
        w[0] -= lr * gw0 / n
        w[1] -= lr * gw1 / n
        b -= lr * gb / n
    print(f"train: fitted for {epochs} epochs, weights={[round(wi, 3) for wi in w]}")
    await asyncio.sleep(2)  # stand-in for a real training loop
    return Model(weights=w, bias=b)
# {{/docs-fragment pipeline}}


# {{docs-fragment evaluate-buggy}}
@env.task
async def evaluate(model: Model, data: Dataset, threshold: float) -> EvaluationReport:
    """Score the model, then register the result with the validation service.

    BUG: `confident_correct / confident_count` raises `ZeroDivisionError` when no prediction
    clears `threshold` (e.g. the release query `--threshold 1.0`), failing the whole run at
    the very last step.
    """
    correct = confident_correct = confident_count = 0
    for x, y in zip(data.features, data.labels):
        p = 1.0 / (1.0 + math.exp(-(model.weights[0] * x[0] + model.weights[1] * x[1] + model.bias)))
        pred = int(p >= 0.5)
        correct += int(pred == y)
        if max(p, 1.0 - p) > threshold:
            confident_count += 1
            confident_correct += int(pred == y)

    accuracy = correct / len(data.labels)
    confident_accuracy = confident_correct / confident_count  # <- ZeroDivisionError when empty

    approved = confident_accuracy >= 0.85
    print(
        f"evaluate: accuracy={accuracy:.3f} confident_accuracy={confident_accuracy:.3f} "
        f"({confident_count} samples above threshold={threshold}) approved={approved}"
    )
    return EvaluationReport(
        accuracy=accuracy,
        confident_accuracy=confident_accuracy,
        confident_count=confident_count,
        approved=approved,
    )
# {{/docs-fragment evaluate-buggy}}


@env.task
async def main(
    n_samples: int = 2000,
    seed: int = 7,
    epochs: int = 5,
    threshold: float = 0.6,
) -> EvaluationReport:
    """Load, preprocess, train, evaluate — the whole nightly pipeline."""
    data = await load_dataset(n_samples, seed)
    clean = await preprocess(data)
    model = await train(clean, epochs)
    return await evaluate(model, clean, threshold)


def summarize(run_name: str) -> None:
    """Print each action's phase. RECOVERED == reused from the source run, never re-executed."""
    for action in Action.listall(for_run_name=run_name):
        print(f"    {action.name:<14} {action.task_name or '-':<28} {action.phase}")


# {{docs-fragment launch-failing-run}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Launch the pipeline as deployed, with the release-evaluation threshold that trips the
    # bug. Everything succeeds until `evaluate` — the very last step — divides by zero.
    #     CLI: flyte run ml_pipeline.py main --threshold 1.0
    run = flyte.run(main, threshold=1.0)
    print(f"seed run: {run.name}\n  {run.url}")
    run.wait(quiet=True)
    print(f"  finished in phase: {run.phase}  (expected: ZeroDivisionError in evaluate)")
    summarize(run.name)
    assert run.phase == ActionPhase.FAILED, "the seed run should fail at evaluate"

    print(
        "\nNext steps:\n"
        "  1. Apply the one-line fix to `evaluate` (see ml_pipeline_fixed.py).\n"
        f"  2. flyte fork {run.name} ml_pipeline.py main --follow\n"
        "Or run the automated tour:  uv run python ml_pipeline_fixed.py"
    )
# {{/docs-fragment launch-failing-run}}
