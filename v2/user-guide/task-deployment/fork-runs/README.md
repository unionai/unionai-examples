# Fork a run — ML pipeline example

The same ML pipeline as the recovery example, but the failure is a bug in our own code, not a
flaky dependency: `evaluate` divides by the number of predictions clearing `threshold`, and a
release evaluation with `--threshold 1.0` yields none — `ZeroDivisionError` at the very last
step. Recovery cannot help (it replays the source run's code as-is); the fix is a code change,
and `flyte fork` is the verb for it.

- `ml_pipeline.py` — the pipeline as deployed, **buggy `evaluate` included**. Running it
  launches the failing run and prints the next steps.
- `ml_pipeline_fixed.py` — the one-line fix, and the automated tour: it launches the buggy
  pipeline, forks the failed run with the fixed code (`flyteplugins.union.fork`), and prints
  which actions were reused.

The two files share `load_dataset`, `preprocess`, `train` and an identical `main`, so the fork
reuses all of them (`RECOVERED` phase) and re-executes only the fixed `evaluate`.

## Run it

Forking is remote-only and ships in `flyteplugins-union` (`pip install flyteplugins-union`).

```bash
# Automated end-to-end tour:
uv run python ml_pipeline_fixed.py

# Or the manual flow — what you would actually do:
flyte run ml_pipeline.py main --threshold 1.0        # fails at evaluate
# ...apply the one-line fix to `evaluate` in ml_pipeline.py...
flyte fork <RUN> ml_pipeline.py main --follow        # reuses load/preprocess/train
flyte get action <FORKED_RUN>                        # reused actions show up as RECOVERED
```

Embedded in the [Fork a run](https://www.union.ai/docs/v2/union/user-guide/tasks/task-deployment/fork-runs/)
user guide page.
