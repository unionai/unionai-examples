# <Tutorial Title>

<!--
Tutorial README template. Copy templates/tutorial/ to v2/tutorials/<name>/ and fill
this in. Keep it a LAB GUIDE — what the tutorial does and how to run it — not
product documentation. If a section starts turning into feature reference (all the
options of TaskEnvironment, every caching mode, the full CLI), move that content to
the product docs on docs.union.ai and link to it instead. See CONTRIBUTING.md.
-->

One-paragraph description: what this tutorial builds and what the reader will learn
by running it. Name the Flyte 2 features it exercises and link to their docs pages
rather than re-explaining them.

## Requirements / spec

- **Inputs**
  - `items: list[str]` — describe each entrypoint parameter and its default.
- **Pipeline**
  1. `step` — what each task does.
  2. `main` — how the entrypoint orchestrates the steps.
- **Outputs**
  - Describe the return value / any artifacts or reports produced.

## Secrets

<!-- Delete this section if the tutorial needs no secrets. -->

| Secret key | Env var | Purpose |
| --- | --- | --- |
| `example-api-key` | `EXAMPLE_API_KEY` | What the key is used for. |

Create secrets with `flyte`/`union` secrets commands (or `test/create_secrets.sh`
for the testing backend). See the [secrets docs](https://docs.union.ai).

## Run it

```bash
# Local (isolated venv, no backend needed):
make test-local FILE=v2/tutorials/<name>/main.py

# Or run directly against a configured backend:
uv run --script main.py
```

## What to look at

Point the reader at the interesting parts of the code (the entrypoint, the key
task, any fragment the docs embed) and at the run output (the run URL, the report).
