# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
make setup-venv          # creates ~/.venv with uv + Python 3.12
source ~/.venv/bin/activate
make update-flyte        # installs latest flyte pre-release into active venv

# Testing (venv must be active)
make test-preview                              # discover scripts without running
make test FILE=v2/tutorials/foo/main.py       # run one script on Union cloud
make test-local FILE=v2/tutorials/foo/main.py # run one script locally (isolated venv per script)
make test FILTER=cognee                        # run all scripts whose path matches "cognee"
make test-local VERBOSE=vv FILE=...           # with flyte verbosity (-v / -vv / -vvv)
make clean                                     # remove test/reports/, test/logs/, test/venvs/

# Direct runner (same flags)
python3 test/test_runner.py --preview
python3 test/test_runner.py --local --file "v2/tutorials/foo/main.py"
```

Cloud tests need `FLYTE_CLIENT_SECRET` set; they target `playground.canary.unionai.cloud` (project `docs-examples`, domain `development`) as configured in `test/config.flyte.yaml`.

## Repository layout

```
v2/         Modern Flyte 2.x examples (only these are tested)
  tutorials/    Full end-to-end tutorial projects (multi-file)
  user-guide/   Short single-file illustrative snippets
v1/         Legacy Flyte 1.x examples (not tested)
_blogs/     Blog-post companion code
test/       Test runner + CI config
```

## Flyte 2.x script conventions

Every runnable script uses **PEP 723 inline metadata** at the top — this is how `uv run` discovers dependencies and how the test runner installs them per-script in isolation:

```python
# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.1.5",
#    "some-lib>=1.0",
# ]
# main = "main"      ← entrypoint task name
# ///
```

### Core Flyte 2.x patterns

**TaskEnvironment** groups config shared across tasks (image, secrets, resources):
```python
env = flyte.TaskEnvironment(
    name="my-env",
    image=flyte.Image.from_uv_script(__file__, name="my-image"),
    secrets=[flyte.Secret(key="my-secret", as_env_var="MY_ENV_VAR")],
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

@env.task
async def my_task(...): ...
```

**`flyte.Image.from_uv_script(__file__)`** builds a container image from the PEP 723 deps automatically. Additional source files needed in the container are added with `.with_source_file(local_path, container_path)`.

**`flyte.io.Dir`** is used to hand off directory state between tasks running in isolated pods — download with `await d.download(local_path=...)`, upload with `await Dir.from_local(local_path, remote_destination=...)`.

**`flyte.map.aio(task_fn, inputs, concurrency=N)`** fans out a task as parallel pods and async-iterates results.

**Scheduling**: `flyte.Trigger` + `flyte.Cron` registers a schedule on the cluster without needing a LaunchPlan.

**`flyte.group("label")`** creates a named span visible in the Union UI execution timeline.

**`report=True`** on a task enables live HTML streaming to the Union UI dashboard.

**`cache="auto"`** keys caching on task inputs + source code hash — useful for idempotent subtasks.

### Deployment

```bash
# Register schedules and deploy apps in one shot
python workflow.py --deploy
python app.py          # also calls flyte.deploy() then flyte.serve()
```

## Active tutorial: cognee_memory_store

The most complex active tutorial. It implements a sleep/wake memory architecture using Cognee (knowledge graph) + Flyte.

**Files:**
- `memory_store.py` — audited, versioned file-based memory store synced via `flyte.io.Dir`. Prefixes enforce access: `reference/` is read-only, `user/` is read-write.
- `agent.py` — staged proposal pipeline: untrusted writes land in `staging/inbox/`, validated, then promoted to `user/`.
- `workflow.py` — all Flyte tasks: `init_reference`, `ingest_url`, `consolidate_cluster`, `sleep_cycle`, `wake_cycle`, `summarize_chat_session`.
- `app.py` — Streamlit UI served via `flyte.app.AppEnvironment`.

**Sleep/wake cycle:**
- **Sleep** (every 6h via Cron): download state → promote staged proposals → cluster + consolidate `user/` memories via `flyte.map.aio` → full graph rebuild (per-topic `empty_dataset` + `cognify`) → upload.
- **Wake** (per question): download state → classify question to topic slugs → `cognee.search(datasets=[slugs])` → Claude answer.
- **`ingest_url`**: scrapes URL via Jina Reader (`r.jina.ai/<url>`) with direct-HTTP fallback, stores in `memory/topic_<slug>/`, `cognify` + `memify` for enrichment.

**Shared object storage paths:**
- `cognee-memory-store/memstore` → memstore files
- `cognee-memory-store/cognee_db` → Cognee SQLite state