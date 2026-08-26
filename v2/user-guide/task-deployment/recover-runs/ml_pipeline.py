# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.6.5",
# ]
# main = "main"
# params = ""
# ///

"""Recover a failed ML training pipeline: reuse every step that succeeded.

A nightly pipeline loads a dataset, preprocesses it, trains a small model, and finishes by
registering the evaluation with an external model-validation service. That service is the
pipeline's only external dependency — and in this story it goes down for maintenance right
when `evaluate` calls it, so the whole run fails at the very last step.

Recovery (`flyte rerun --recover` / `flyte.rerun(..., recover=True)`) relaunches the run with
its original code and inputs, reusing every action that already succeeded: the dataset is not
re-read, the features are not recomputed, the model is not retrained. Only `evaluate`
re-executes. Once the validation service is back, the recovered run completes.

The outage is simulated with the `VALIDATION_SERVICE_OUTAGE` environment variable so the
failure is reproducible; recovery clears it with an `-e` / `env_vars` override. Because the
source run's environment is inherited by the recovered run, the override is what models "the
service has recovered".

Then a second twist: the team decides the evaluation threshold should have been 0.9, not 0.6.
`threshold` is a root input consumed only by `evaluate`, so a recovery with the changed input
re-executes `evaluate` and nothing else — load, preprocess and train are reused as-is.

Run the whole tour (remote only — rerun and recover are not supported locally):

    uv run python ml_pipeline.py

--------------------------------------------------------------------------------------------
Equivalent `flyte` CLI commands
--------------------------------------------------------------------------------------------
The script prints the seed run's name; substitute it for <RUN> below.

    # Seed run: launched during the simulated validation-service outage; fails at `evaluate`.
    flyte run ml_pipeline.py main -e VALIDATION_SERVICE_OUTAGE=1

    # 1. Recover: the outage is over, so clear the switch. Everything that succeeded in the
    #    seed run is reused (RECOVERED phase); only `evaluate` re-executes.
    flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --follow

    # 2. Recover AND change an input. `threshold` feeds only `evaluate`, so the upstream steps
    #    keep their recorded outputs and only `evaluate` re-executes against the new value.
    flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --threshold 0.9 --follow

    # Inspect what recovery reused — reused actions show up as RECOVERED:
    flyte get action <RECOVERED_RUN>
"""

import asyncio
import math
import os
import random
from dataclasses import dataclass

import flyte
from flyte.models import ActionPhase
from flyte.remote import Action

env = flyte.TaskEnvironment(
    name="ml_pipeline",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)

#: Set to "1" on a run to simulate the external validation service being down.
VALIDATION_SERVICE_OUTAGE = "VALIDATION_SERVICE_OUTAGE"


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


# {{docs-fragment evaluate}}
@env.task
async def evaluate(model: Model, data: Dataset, threshold: float) -> EvaluationReport:
    """Score the model, then register the result with the external validation service.

    The service call is the pipeline's final step — an intermittent 503 here fails the whole
    run after everything expensive has already finished. That is exactly the shape recovery
    is for.
    """
    correct = confident_correct = confident_count = 0
    for x, y in zip(data.features, data.labels):
        p = 1.0 / (1.0 + math.exp(-(model.weights[0] * x[0] + model.weights[1] * x[1] + model.bias)))
        pred = int(p >= 0.5)
        correct += int(pred == y)
        if max(p, 1.0 - p) >= threshold:
            confident_count += 1
            confident_correct += int(pred == y)

    accuracy = correct / len(data.labels)
    confident_accuracy = confident_correct / confident_count if confident_count else 0.0

    # The simulated outage: during the maintenance window the call fails with a 503.
    # In a real pipeline this would be an ordinary HTTP call to a flaky upstream service.
    if os.environ.get(VALIDATION_SERVICE_OUTAGE) == "1":
        raise ConnectionError(
            "HTTP 503 Service Unavailable: POST https://validation.internal/api/v1/approve"
        )

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
# {{/docs-fragment evaluate}}


# {{docs-fragment main}}
@env.task
async def main(
    n_samples: int = 2000,
    seed: int = 7,
    epochs: int = 5,
    threshold: float = 0.6,
) -> EvaluationReport:
    """Load, preprocess, train, evaluate. `threshold` feeds only the final step."""
    data = await load_dataset(n_samples, seed)
    clean = await preprocess(data)
    model = await train(clean, epochs)
    return await evaluate(model, clean, threshold)
# {{/docs-fragment main}}


def summarize(run_name: str) -> None:
    """Print each action's phase. RECOVERED == reused from the source run, never re-executed."""
    for action in Action.listall(for_run_name=run_name):
        print(f"    {action.name:<14} {action.task_name or '-':<28} {action.phase}")


# {{docs-fragment recover-tour}}
if __name__ == "__main__":
    flyte.init_from_config()

    # --- The seed run: launched during the simulated outage, fails at `evaluate`. ----------
    #     CLI: flyte run ml_pipeline.py main -e VALIDATION_SERVICE_OUTAGE=1
    seed = flyte.with_runcontext(env_vars={VALIDATION_SERVICE_OUTAGE: "1"}).run(main)
    print(f"seed run: {seed.name}\n  {seed.url}")
    seed.wait(quiet=True)
    print(f"  finished in phase: {seed.phase}  (expected: the validation service is 'down')")
    summarize(seed.name)
    assert seed.phase == ActionPhase.FAILED, "the seed run should fail at evaluate"

    # --- 1. Recover: the outage is over, so recover the run with the switch cleared. --------
    #     Every succeeded action of the seed run is reused as-is — the dataset is not re-read,
    #     the features are not recomputed, the model is not retrained. Only `evaluate`
    #     re-executes, and this time the service answers.
    #     CLI: flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --follow
    recovered = flyte.with_runcontext(env_vars={VALIDATION_SERVICE_OUTAGE: "0"}).rerun(
        seed.name, recover=True
    )
    print(f"\n1. rerun(recover=True) -> {recovered.name}\n  {recovered.url}")
    recovered.wait()
    print(f"  finished in phase: {recovered.phase}")
    print("  (load_dataset / preprocess / train should read RECOVERED; evaluate re-executed)")
    summarize(recovered.name)
    assert recovered.phase == ActionPhase.SUCCEEDED, "recovery should complete the pipeline"

    # --- 2. Recover with a changed input: the eval threshold should have been 0.9. ----------
    #     `threshold` is consumed only by `evaluate`, so the upstream actions keep the outputs
    #     they produced under the original inputs and only `evaluate` re-executes. That is the
    #     safe case for changed inputs — no recovered output goes stale, so nothing needs
    #     `--force-rerun-action`.
    #     CLI: flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --threshold 0.9 --follow
    regraded = flyte.with_runcontext(env_vars={VALIDATION_SERVICE_OUTAGE: "0"}).rerun(
        seed.name, recover=True, threshold=0.9
    )
    print(f"\n2. rerun(recover=True, threshold=0.9) -> {regraded.name}\n  {regraded.url}")
    regraded.wait()
    print(f"  finished in phase: {regraded.phase}")
    print("  (upstream RECOVERED again — the changed input only feeds `evaluate`)")
    summarize(regraded.name)
    assert regraded.phase == ActionPhase.SUCCEEDED
# {{/docs-fragment recover-tour}}
