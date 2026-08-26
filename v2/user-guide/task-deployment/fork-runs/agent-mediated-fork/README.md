# Agent-mediated forking

An agent, running as a Flyte task, develops and debugs another Flyte workflow — no human in the
loop.

- `workflow.py` — a toy data-processing pipeline (`load_records` → `clean_records` →
  `summarize`) over mock sales data, shipped with two planted bugs (a wrong field name, a wrong
  dict access), one per downstream step.
- `agent_task.py` — the agent. It launches `workflow.main`, reads the failed action's error,
  picks a patch from its playbook, edits `workflow.py` on the pod, reloads it, and forks the
  failed run with the fixed code. It repeats until the workflow succeeds.

Because each iteration **forks** the latest failed run, every action that already succeeded is
reused: `load_records` executes exactly once across the whole debug session, and each fix
re-executes only the step it repairs. That is the point of the example — fork turns the
edit-run-observe debugging loop into something an agent can run cheaply inside the platform.

`propose_fix` is a deterministic playbook so the example runs anywhere; in a real system that
function is an LLM call (source + error message in, patch out). Forking does not care which
produced the edit.

## Run it

Forking is remote-only, so the full story needs a Union backend:

```bash
uv run python agent_task.py
```

Watch the agent's logs: attempt 0 fails in `clean_records`, iteration 1 patches the field name
and forks (reusing `load_records`), iteration 2 patches the summary and forks again (reusing
`load_records` *and* `clean_records`), and the workflow succeeds.

Running locally (`make test-local`) exercises the same loop without a control plane — the
agent re-runs the patched workflow inline instead of forking.

> The agent edits code that it then executes. Sandbox and review before pointing this pattern
> at real code.
