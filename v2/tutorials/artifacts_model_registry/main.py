# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.12.2",
# ]
# main = "main"
# params = ""
# ///

# Artifacts as a model registry.
#
# Stages are partitions on one artifact name (`stage=candidate`, `stage=production`),
# promotion publishes a new version into the production partition with lineage
# back to the candidate, and two partition-scoped triggers gate evaluation and
# rollout. A sweep task doubles as the experiment: each `train` child action is a
# tracked run, the parent renders the leaderboard as its report and attaches it to
# the candidate as a card.
#
# Tutorial: https://docs.union.ai/tutorials/model-training/artifacts-model-registry

import asyncio
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import flyte
import flyte.artifacts as artifacts
import flyte.report
from flyte.io import Dir, File
from flyte.remote import Artifact

# {{docs-fragment env}}
MODEL = "sentiment-model"
DATASET = "reviews"

env = flyte.TaskEnvironment(
    name="model_registry",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)
# {{/docs-fragment env}}


# {{docs-fragment train}}
@dataclass
class TrainResult:
    lr: float
    accuracy: float
    f1: float
    weights: Dir


@env.task(cache="auto")
async def train(dataset: Artifact, lr: float) -> TrainResult:
    """One run of the experiment. Cached on (dataset version, lr), so re-running
    the sweep with the same inputs does not retrain."""
    data = await dataset.to_python.aio(File)
    async with data.open("rb") as fh:
        n = len((await fh.read()).splitlines())
    out = Path(tempfile.mkdtemp())
    (out / "weights.safetensors").write_text(f"lr={lr} rows={n}")
    (out / "config.json").write_text('{"architecture": "distilbert-base"}')
    weights = await Dir.from_local(out)
    # Stand-in metrics that depend on lr so the leaderboard has an order.
    accuracy = round(0.9 + lr * 10, 4)
    return TrainResult(lr=lr, accuracy=accuracy, f1=round(accuracy - 0.02, 4), weights=weights)


# {{/docs-fragment train}}


# {{docs-fragment sweep}}
@env.task(report=True)
async def sweep(dataset: Artifact, lrs: list[float]) -> Artifact:
    """The experiment. One `train` action per lr, the comparison as this action's
    report, and the winner published as a candidate."""
    results = await asyncio.gather(*(train(dataset, lr) for lr in lrs))
    rows = sorted(results, key=lambda r: -r.accuracy)
    table = leaderboard_html(rows)
    await flyte.report.log.aio(table, do_flush=True)

    best = rows[0]
    card = await artifacts.Card.create_from.aio(
        content=table, format="html", card_type="model"
    )
    return await Artifact.create.aio(
        best.weights,
        name=MODEL,
        kind="model",
        description=f"sweep winner lr={best.lr}",
        attrs={
            "framework": "torch",
            "architecture": "distilbert-base",
            "task": "text-classification",
            "serial_format": "safetensors",
            "accuracy": str(best.accuracy),
            "f1": str(best.f1),
            "lr": str(best.lr),
        },
        card=card,
        partitions={"stage": "candidate", "date": date.today()},
        # This model version derives from that dataset version.
        parents=[dataset.artifact_version_id],
    )


# {{/docs-fragment sweep}}


# {{docs-fragment on-candidate}}
on_candidate = flyte.Trigger(
    name="eval-candidate",
    # Only candidate versions fire this; promotions never do.
    automation=flyte.OnArtifact(MODEL, stage="candidate"),
    inputs={"candidate": flyte.TriggeredArtifact},
    description="Evaluate every new candidate against production",
)
# {{/docs-fragment on-candidate}}


# {{docs-fragment evaluate}}
@env.task(triggers=[on_candidate])
async def evaluate(candidate: Artifact) -> str:
    cand_acc = float(attrs_of(candidate)["accuracy"])
    try:
        prod = await Artifact.get.aio(MODEL, stage="production")
        prod_acc = float(attrs_of(prod)["accuracy"])
    except Exception:  # nothing promoted yet
        prod_acc = -1.0

    if cand_acc <= prod_acc:
        return f"rejected {candidate.version}: accuracy {cand_acc} <= production {prod_acc}"

    # Promote: republish the same value into the production partition. No blob is
    # copied, and parents=[candidate] records what this version was promoted from.
    weights = await candidate.to_python.aio(Dir)
    promoted = await Artifact.create.aio(
        weights,
        name=MODEL,
        kind="model",
        description=f"promoted from {candidate.version}",
        attrs={**attrs_of(candidate), "promoted_from": candidate.version},
        partitions={"stage": "production", "date": date.today()},
        parents=[candidate.artifact_version_id],
    )
    return f"promoted {candidate.version} -> {promoted.version}"


# {{/docs-fragment evaluate}}


# {{docs-fragment on-production}}
on_production = flyte.Trigger(
    name="prod-rollout",
    # Only production versions fire this.
    automation=flyte.OnArtifact(MODEL, stage="production"),
    inputs={"model": flyte.TriggeredArtifact},
    # Stay on a released version of `rollout` while newer ones are deployed.
    # Change the pin and redeploy to move the production path.
    task_version=os.environ.get("ROLLOUT_TASK_VERSION", "v1.4.0"),
    description="Hand a promoted model to serving",
)
# {{/docs-fragment on-production}}


# {{docs-fragment rollout}}
@env.task(triggers=[on_production])
async def rollout(model: Artifact) -> str:
    """Smoke-test the promoted weights and hand off to serving.

    A real rollout redeploys the serving app with
    `flyte.app.ArtifactValue(name=MODEL, version=model.version)` as a parameter.
    The resolved version is recorded on the app, so the app answers
    "what is production" as well as the `stage=production` partition does.
    """
    weights = await model.to_python.aio(Dir)
    config = await weights.get_file("config.json")
    assert config is not None, "promoted model is missing config.json"
    return model.version


# {{/docs-fragment rollout}}


# {{docs-fragment main}}
@env.task
async def main(lrs: list[float] = [1e-3, 3e-4, 1e-4]) -> str:
    """Register a dataset and run one sweep. The triggers take it from there:
    candidate -> evaluate -> production -> rollout."""
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("text,label\ngreat,1\nterrible,0\nfine,1\n")
    dataset = await Artifact.create.aio(
        await File.from_local(f.name),
        name=DATASET,
        kind="data",
        external_ref="s3://corp-data/reviews/2026-11.csv",
    )
    candidate = await sweep(dataset, lrs)
    return candidate.version


# {{/docs-fragment main}}


def attrs_of(a: Artifact) -> dict[str, str]:
    return dict(a.pb2.spec.info.user_metadata)


def leaderboard_html(rows: list[TrainResult]) -> str:
    body = "".join(
        f"<tr><td>{r.lr}</td><td>{r.accuracy:.4f}</td><td>{r.f1:.4f}</td></tr>" for r in rows
    )
    return (
        f"<h2>{MODEL} sweep</h2>"
        f"<table><tr><th>lr</th><th>accuracy</th><th>f1</th></tr>{body}</table>"
    )


# {{docs-fragment query}}
def show_registry() -> None:
    """Browse the registry the way you would a model registry UI."""
    print("candidates:")
    for a in Artifact.listall(MODEL, stage="candidate", limit=10):
        print(f"  {a.version}  accuracy={attrs_of(a).get('accuracy')}  {a.source_run_url}")
    prod = Artifact.get(MODEL, stage="production")
    print(f"production: {prod.version} (promoted from {attrs_of(prod).get('promoted_from')})")


# {{/docs-fragment query}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.with_runcontext(labels={"experiment": "lr-sweep"}).run(main)
    print(run.name)
    print(run.url)
    run.wait()
    show_registry()
