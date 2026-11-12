# Artifacts as a model registry

A sweep publishes its best model as a `stage=candidate` artifact version, a trigger
evaluates it against production, promotion republishes it into the
`stage=production` partition with lineage back to the candidate, and a second
trigger pinned to a released task version hands it to serving. No mutable alias is
involved: every transition is a new artifact version, and every version records
what it was promoted from.

Tutorial with the full walkthrough:
[Artifacts as a model registry](https://docs.union.ai/tutorials/model-training/artifacts-model-registry).

## Requirements / spec

- **Inputs**
  - `lrs: list[float]` — learning rates to sweep. Default `[1e-3, 3e-4, 1e-4]`.
- **Pipeline**
  1. `main` — registers a small dataset artifact and runs one `sweep`.
  2. `sweep` — one cached `train` action per lr, a leaderboard report, and the
     winner published as `sentiment-model` with `stage=candidate`.
  3. `evaluate` — fired by `on_candidate`. Compares the candidate's accuracy with
     the current `stage=production` version and, if better, publishes the same
     weights as a new `stage=production` version with `parents=[candidate]`.
  4. `rollout` — fired by `on_production`. Smoke-tests the promoted weights.
     The trigger carries `task_version`, so it stays on a released version of
     `rollout` while newer ones are deployed.
- **Outputs**
  - Artifact versions of `sentiment-model` in the `candidate` and `production`
    partitions, each with metrics in `attrs` and a leaderboard card.
  - A report on the `sweep` action.

## Run it

Deploy the environment so the two triggers exist. `rollout`'s trigger is pinned to
`v1.4.0`, so deploy under that version once, or point `ROLLOUT_TASK_VERSION` at a
version you have deployed:

```bash
cd v2/tutorials/artifacts_model_registry
flyte deploy --version v1.4.0 main.py env
uv run --script main.py
```

`main` returns the candidate's version. Within a minute an `evaluate` run appears,
launched by the trigger, followed by a `rollout` run if the candidate was promoted.

## What to look at

- The `sweep` action's report: the leaderboard across the `train` actions.
- The `sentiment-model` artifact in the UI: two partitions, the candidate's card,
  and the production version's lineage pointing at the candidate and at the
  `reviews` dataset.
- `show_registry()` at the bottom of `main.py`: the same questions asked from
  Python with `flyte.remote.Artifact.listall` and `flyte.remote.Artifact.get`.
