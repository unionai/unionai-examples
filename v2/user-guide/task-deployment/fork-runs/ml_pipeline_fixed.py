# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.6.5",
#    "flyteplugins-union>=0.8.1",
# ]
# main = "main"
# params = ""
# ///

"""The same ML pipeline, one commit later: `evaluate` fixed, ready to fork.

This is the state of the code after the one-line fix to `evaluate` from `ml_pipeline.py`:

    - confident_accuracy = confident_correct / confident_count
    + confident_accuracy = confident_correct / confident_count if confident_count else 0.0

`load_dataset`, `preprocess` and `train` are imported unchanged from `ml_pipeline.py`, and
`main`'s body is identical, so forking a failed run of `ml_pipeline.py` with this file's
`main` reuses every one of their actions and re-executes only `evaluate`:

    # Manual flow — fork the failed run with your edited working tree:
    flyte fork <RUN> ml_pipeline.py main --follow          # after editing ml_pipeline.py
    flyte fork <RUN> ml_pipeline_fixed.py main --follow    # or fork this file directly

    # Programmatic flow — same thing, from Python:
    from flyteplugins.union import fork
    forked = fork("<RUN>", task_template=main)

Run the automated end-to-end tour (remote only — forking has no local equivalent):

    uv run python ml_pipeline_fixed.py            # launches the buggy run, then forks it
    uv run python ml_pipeline_fixed.py <RUN>      # ...or fork a run you launched yourself
"""

import math
import sys

import flyte
from flyte.models import ActionPhase
from flyte.remote import Action
from flyteplugins.union import fork

# {{docs-fragment fixed-code}}
# The upstream steps are unchanged by the fix — importing them guarantees their task
# identities are exactly the ones the failed run recorded, so a fork reuses them as-is.
from ml_pipeline import (
    Dataset,
    EvaluationReport,
    Model,
    env,
    load_dataset,
    preprocess,
    train,
)


@env.task
async def evaluate(model: Model, data: Dataset, threshold: float) -> EvaluationReport:
    """Score the model, then register the result with the validation service.

    FIXED: an empty confident subset reports a confident accuracy of 0.0 instead of raising
    `ZeroDivisionError`.
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
    confident_accuracy = (
        confident_correct / confident_count
        if confident_count
        else 0.0  # <- the fix
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
# {{/docs-fragment fixed-code}}


def summarize(run_name: str) -> None:
    """Print each action's phase. RECOVERED == reused from the source run, never re-executed."""
    for action in Action.listall(for_run_name=run_name):
        print(f"    {action.name:<14} {action.task_name or '-':<28} {action.phase.value}")


# {{docs-fragment fork-tour}}
if __name__ == "__main__":
    flyte.init_from_config()

    if len(sys.argv) > 1:
        # Fork a run you launched yourself (e.g. via `flyte run ml_pipeline.py main --threshold 1.0`).
        source_name = sys.argv[1]
    else:
        # Automated tour: first launch the buggy pipeline exactly as it was deployed.
        # A threshold of 1.0 trips the ZeroDivisionError in `evaluate`.
        import ml_pipeline

        seed = flyte.run(ml_pipeline.main, threshold=1.0)
        print(f"seed run (buggy code): {seed.name}\n  {seed.url}")
        seed.wait(quiet=True)
        print(f"  finished in phase: {seed.phase.value}  (expected: ZeroDivisionError in evaluate)")
        summarize(seed.name)
        assert seed.phase == ActionPhase.FAILED, "the seed run should fail at evaluate"
        source_name = seed.name

    # Fork the failed run with this file's code: the fix to `evaluate` is the only change, so
    # load_dataset / preprocess / train are reused and only `evaluate` re-executes.
    #     CLI: flyte fork <RUN> ml_pipeline_fixed.py main --follow
    forked = fork(source_name, task_template=main)
    print(f"\nfork {source_name} -> {forked.name}\n  {forked.url}")
    forked.wait()
    print(f"  finished in phase: {forked.phase.value}")
    print("  (load_dataset / preprocess / train should read RECOVERED; evaluate re-executed)")
    summarize(forked.name)
    assert forked.phase == ActionPhase.SUCCEEDED, "the forked run should complete"
# {{/docs-fragment fork-tour}}
