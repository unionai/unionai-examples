# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.6.5",
#    "flyteplugins-union>=0.8.1",
# ]
# main = "agent"
# params = ""
# ///

"""Agent-mediated forking: an agent task that develops and debugs a Flyte workflow.

`workflow.py` next to this file is a toy data-processing pipeline that ships with bugs.
Instead of a human in the debug loop, the `agent` task below closes the loop itself, from
inside a Flyte pod:

    1. Launch `workflow.main` as a real run and observe it fail.
    2. Read the failed action's error message and pick a patch from its playbook.
    3. Apply the patch to `workflow.py` on disk and reload the module.
    4. `fork()` the failed run with the fixed code — reusing every action that already
       succeeded, so only the broken step (and anything after it) re-executes.
    5. Repeat until the run succeeds.

That is the power of forking: each fix builds on the previous run's completed work instead of
re-running the whole pipeline. Across this example's two bugs, `load_records` executes exactly
once — every fix forks from the latest failed run.

The "brain" here (`propose_fix`) is a deterministic playbook so the example runs anywhere. In
a real system it is an LLM call — feed it the workflow source and the failed action's error
message, and let it return the patch (see the agent tutorials under `v2/tutorials/`). Forking
does not care which produced the edit.

Notes:

  * Forking is remote-only. When this example runs locally (no control plane), the agent falls
    back to re-running the workflow inline after each patch — same loop, no action reuse.
  * The agent edits code that it then runs. Treat this pattern with the sandboxing and review
    it deserves before pointing it at real code.

Run it (remote, the full story):

    uv run python agent_task.py
"""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import flyte
from flyte.models import ActionPhase
from flyte.remote import Action
from flyteplugins.union import fork

# {{docs-fragment imports-and-env}}
# Importing `workflow` here does two things: it makes `workflow.py` a loaded module, so it
# ships in this file's code bundle, and it gives the agent its first handle on the pipeline.
import workflow  # noqa: F401

env = flyte.TaskEnvironment(
    name="workflow_agent",
    resources=flyte.Resources(cpu=1, memory="500Mi"),
)

WORKFLOW_FILE = Path(__file__).with_name("workflow.py")
# {{/docs-fragment imports-and-env}}


# {{docs-fragment playbook}}
# The agent's playbook: (error signature, buggy snippet, fixed snippet). `propose_fix` picks
# the patch whose signature matches the observed failure; a deterministic stand-in for an LLM.
KNOWN_BUGS = [
    ("KeyError", 'record["price"]', 'record["unit_price"]'),
    ("TypeError", 'kv[1]["total"]', 'kv[1]'),
]


def propose_fix(source: str, error_message: str) -> tuple[str, str] | None:
    """Pick the next patch: one that matches the error, else the first still-applicable one."""
    applicable = [(old, new) for _, old, new in KNOWN_BUGS if old in source]
    if not applicable:
        return None
    for hint, old, new in KNOWN_BUGS:
        if old in source and hint.lower() in error_message.lower():
            return old, new
    return applicable[0]
# {{/docs-fragment playbook}}


def apply_patch(source: str, old: str, new: str) -> tuple[str, ModuleType]:
    """Patch the workflow source and reload the module so the fix takes effect.

    The fix is written to disk first: code bundles are built from the working tree, and child
    actions re-import the module from disk, so the fix only counts once the file itself has
    changed. In local runs the agent restores the file when it is done.
    """
    patched = source.replace(old, new)
    WORKFLOW_FILE.write_text(patched, encoding="utf-8")
    module = importlib.reload(sys.modules["workflow"])
    return patched, module


def _control_plane_available() -> bool:
    """True in remote task pods, where runs can be launched and forked."""
    try:
        from flyte._initialize import get_client

        return get_client() is not None
    except Exception:
        return False


async def _first_failure(run_name: str) -> tuple[str, str]:
    """The pipeline step that failed a run, and its error message."""
    fallback: tuple[str, str] = ("unknown", "")
    async for action in Action.listall.aio(for_run_name=run_name, in_phase=(ActionPhase.FAILED,)):
        details = await action.details()
        message = details.error_info.message if details.error_info else ""
        if action.parent_name:  # a step inside the pipeline, not the root action
            return action.task_name or action.name, message
        if message and not fallback[1]:
            fallback = (action.task_name or action.name, message)
    return fallback


# {{docs-fragment agent}}
@env.task
async def agent(n_records: int = 50, seed: int = 7, max_iterations: int = 5) -> dict:
    """Run the develop-debug loop until `workflow.main` succeeds. Returns a report."""
    remote = _control_plane_available()
    original_source = WORKFLOW_FILE.read_text(encoding="utf-8")
    source = original_source
    wf = sys.modules["workflow"]
    attempts: list[dict] = []
    phase = ActionPhase.FAILED
    run = None

    # --- Attempt 0: run the workflow as written, and observe how it fails. ----------------
    if remote:
        run = await flyte.run.aio(wf.main, n_records=n_records, seed=seed)
        print(f"[attempt 0] launched {run.name}: {run.url}")
        await run.wait.aio(quiet=True)
        phase = run.phase
        failed_task, error = (
            (None, "") if phase == ActionPhase.SUCCEEDED else await _first_failure(run.name)
        )
    else:
        try:
            await wf.main(n_records=n_records, seed=seed)
            phase = ActionPhase.SUCCEEDED
            failed_task, error = None, ""
        except Exception as e:
            failed_task, error = "workflow", f"{type(e).__name__}: {e}"
            print(f"[attempt 0] failed: {error}")

    # --- The loop: observe -> patch -> fork (or re-run locally) -> observe ... -----------
    try:
        for iteration in range(1, max_iterations + 1):
            if phase == ActionPhase.SUCCEEDED:
                break
            if failed_task is None:
                raise RuntimeError("the workflow failed, but no failed step was found")

            patch = propose_fix(source, error or "")
            if patch is None:
                raise RuntimeError(f"the playbook has no patch left for: {error!r}")
            old, new = patch
            print(f"[iteration {iteration}] {failed_task} failed with: {error}")
            print(f"[iteration {iteration}] patching {old!r} -> {new!r}")
            source, wf = apply_patch(source, old, new)
            attempts.append({"failed_task": failed_task, "error": error, "patch": f"{old} -> {new}"})

            if remote:
                # Fork the run that just failed: every action that succeeded in it is reused,
                # and only the patched step (and anything downstream) re-executes.
                run = await fork.aio(run.name, task_template=wf.main)
                print(f"[iteration {iteration}] forked {run.name}: {run.url}")
                await run.wait.aio(quiet=True)
                phase = run.phase
                failed_task, error = (
                    (None, "") if phase == ActionPhase.SUCCEEDED else await _first_failure(run.name)
                )
            else:
                # Local mode has no control plane to fork from — re-run the patched workflow.
                try:
                    await wf.main(n_records=n_records, seed=seed)
                    phase = ActionPhase.SUCCEEDED
                    failed_task, error = None, ""
                except Exception as e:
                    failed_task, error = "workflow", f"{type(e).__name__}: {e}"
    finally:
        if not remote:
            # Leave the working tree as we found it: local patches are ephemeral.
            WORKFLOW_FILE.write_text(original_source, encoding="utf-8")

    if phase != ActionPhase.SUCCEEDED:
        raise RuntimeError(f"the workflow still fails after {max_iterations} iterations: {error}")

    print(f"[done] the workflow succeeds after {len(attempts)} patch(es)")
    return {
        "mode": "remote: fork reused every succeeded action" if remote else "local: fresh re-runs (fork is remote-only)",
        "final_run": run.name if run is not None else "(local)",
        "patches": json.dumps(attempts),
        "iterations": str(len(attempts)),
    }
# {{/docs-fragment agent}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(agent)
    print(run.name)
    print(run.url)
    run.wait()
    print(run.outputs())
