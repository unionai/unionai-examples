# AutoTrain

An AutoML web app on Union: a FastAPI frontend kicks off a multi-agent training
pipeline (data profiling → experiment design → research/training loop) as a
Flyte run. This is a [hybrid app–task graph](https://www.union.ai/docs/v2/union/user-guide/build-apps/hybrid-graphs/):
the app and the pipeline tasks run on the same cluster, and the app submits
runs with `flyte.run.aio()`.

## Layout

| Path | Purpose |
|---|---|
| `app.py` | FastAPI frontend + `FastAPIAppEnvironment` (web image only) |
| `pipeline.py` | Task environments (cpu/gpu images, secrets) and the `automl_pipeline` task |
| `agents/` | Data / design / research agents (imported lazily inside tasks) |
| `templates/` | Jinja2 pages for the form and the status page |

The design agent pushes an experiment branch to a GitHub repo; the research
agent clones that branch, runs training experiments, and opens a PR with
results. The branch is the design→research handoff, so a **writable** GitHub
token is required even for public repos.

## Prerequisites

1. **Flyte config** (`~/.flyte/config.yaml`) pointing at the target cluster
   (currently `demo.hosted.unionai.cloud`, org `demo`).
2. **Secrets** in the project/domain you deploy to (`flytesnacks/development`):

   ```sh
   flyte create secret --project flytesnacks --domain development internal-anthropic-api-key <key>
   flyte create secret --project flytesnacks --domain development autotrain-github-token <PAT>
   ```

   The GitHub token must be able to push and open PRs on the experiments repo.
   For a fine-grained PAT: Repository access = *Only select repositories*
   (the "Public repositories" option is read-only), **Contents: Read and
   write**, **Pull requests: Read and write**. For an org-owned repo the token's
   resource owner must be the org, and org policy may require admin approval —
   an unapproved or expired token fails with the same 401 as a bad one.

   Task pods fail at admission with
   `flyte-pod-webhook ... failed to inject secret` when a referenced secret
   does not exist in the target project/domain.

3. `uv` (the deploy command below uses `uvx`).

## Deploy

```sh
uvx --with fastapi --with uvicorn --with jinja2 --with python-multipart \
    --with "connectrpc==0.10.1" --from "flyte==2.5.9" \
    flyte serve --project flytesnacks --domain development app.py automl_webapp
```

Why the extra flags:

- `flyte serve` imports `app.py`, so the CLI environment needs the app's web
  dependencies (fastapi, uvicorn, jinja2, python-multipart).
- `connectrpc==0.10.1` is pinned because `connectrpc 0.11.0` breaks
  `flyte 2.5.x` uploads with `'Headers' object is not callable`.

Deploying only builds the **web** image. The cpu/gpu task images are built
remotely on the first run that needs them (~5–7 min for the gpu image; cached
afterwards). The app scales to zero when idle — the first request after a
quiet period takes ~30–60 s.

Local dev server (UI only; submissions go to the cluster via your local
config): `python app.py --local`.

## Using it

Open the app endpoint (`flyte get app --project flytesnacks --domain
development` shows the URL) and submit the form, or POST directly:

```sh
curl -i -X POST https://<app-endpoint>/run \
  -d "dataset_link=https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" \
  -d "target_column=Survived" \
  -d "github_repo=unionai-oss/autoresearch-experiments" \
  -d "domain=auto" -d "max_experiments=1" -d "time_budget=60" -d "max_samples_raw=100"
```

The 303 redirect points at `/status/{job_id}`; `/api/status/{job_id}` returns
JSON. If `WEBHOOK_API_KEY` is set at deploy time, `/run` requires
`Authorization: Bearer <key>`.

## How run tracking works (stateless)

The app keeps **no** job state. Each submission becomes a run named
`automl-<job_id>` (job ids are 20 hex chars — run names are capped at
**30 characters**) with labels `app=automl-webapp` and `automl-job-id=<job_id>`.
The status endpoint resolves everything from the cluster: phase → status,
run outputs → result, run details → error message. Status therefore survives
app restarts and redeploys. The one exception: a submission that fails before
a run exists is held in a small in-process error map so the status page can
show the traceback (lost on restart — check app logs in the console if a job
sits in "starting").

## Debugging

```sh
flyte get run    --project flytesnacks --domain development                 # recent runs
flyte get action --project flytesnacks --domain development <run>          # actions + errorInfo
flyte get logs   --project flytesnacks --domain development <run> <action> # task logs
```

Failure modes seen in practice:

| Symptom | Cause / fix |
|---|---|
| Status page stuck on "Building container images…" | Normal on first run (gpu image is slow). If no run ever appears: the submission failed — status page shows the traceback; also check app logs. |
| `Event loop stopped before Future completed` | `flyte.run.aio` driven from a thread with its own event loop. It must be awaited on the loop where flyte was initialized (`asyncio.create_task` in the endpoint). |
| `project_id.organization: must be at least 1 characters` | The app re-initialized flyte in-cluster. The `flyte serve` runtime already initializes (org comes from its `--org` flag); re-running `init_in_cluster()` drops the org. Only init when `flyte._initialize.is_initialized()` is false. |
| `run_id.name: must be at most 30 characters` | Run names are limited to 30 chars — keep `automl-` + job id within that. |
| Pod admission: `failed to inject secret` | A secret in `pipeline.py` doesn't exist in the target project/domain. |
| `git push` fails: `Invalid username or token` / password prompt | Token missing, expired, unapproved by the org, or lacking Contents write. Also: tokens must be embedded as `https://x-access-token:<token>@github.com/...` — a bare token-as-username fails for fine-grained PATs. |
| Deploy fails: `'Headers' object is not callable` | `connectrpc>=0.11` with `flyte 2.5.1` — pin `connectrpc==0.10.1`. |
