# Recover a failed run — ML pipeline example

A nightly ML pipeline (load → preprocess → train → evaluate) that fails at the very last step
because the external model-validation service is down for maintenance — simulated with the
`VALIDATION_SERVICE_OUTAGE` env var so the failure is reproducible.

`ml_pipeline.py` runs the whole tour against a Union backend:

1. **The failure.** The run is launched during the simulated outage and fails at `evaluate`,
   after the expensive steps have all succeeded.
2. **Recover.** `flyte.rerun(run, recover=True)` (CLI: `flyte rerun <run> --recover`) relaunches
   the run with the outage switch cleared. `load_dataset`, `preprocess` and `train` land in the
   `RECOVERED` phase — reused, never re-executed — and only `evaluate` runs. The pipeline
   completes.
3. **Recover with changed inputs.** The eval threshold should have been `0.9`. Since `threshold`
   feeds only `evaluate`, `flyte.rerun(run, recover=True, threshold=0.9)` re-executes only
   `evaluate`; every upstream action is reused as-is.
4. **Recover with a forced re-run.** `train` succeeded in the source run, so recovery would
   reuse it. Naming it in `force_rerun_actions` re-executes it anyway — handy when you want a
   fresh model before re-evaluating, and it re-enqueues its children too.

## Run it

Recovery is remote-only — you need a configured Union backend (`flyte.init_from_config()`).

```bash
# End-to-end tour (all four phases):
uv run python ml_pipeline.py

# Or step by step, from the CLI:
flyte run ml_pipeline.py main -e VALIDATION_SERVICE_OUTAGE=1
flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --follow
flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --threshold 0.9 --follow
flyte get action <RUN>     # copy the train action's name
flyte rerun <RUN> --recover -e VALIDATION_SERVICE_OUTAGE=0 --threshold 0.9 \
    --force-rerun-action <TRAIN_ACTION> --follow
flyte get action <RECOVERED_RUN>     # reused actions show up as RECOVERED
```

Embedded in the [Recover a run](https://www.union.ai/docs/v2/union/user-guide/tasks/task-deployment/recover-runs/)
user guide page.
