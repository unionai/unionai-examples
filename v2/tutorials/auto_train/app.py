"""
AutoTrain — FastAPI frontend app.

Deploy to Union (builds only the web image):
    flyte serve app.py automl_webapp

Local dev server:
    python app.py --local

Pipeline task images (cpu/gpu) are built separately when training is first submitted.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import asyncio

import fastapi
from fastapi import Form, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette import status

import flyte
import flyte.remote
from flyte.app.extras import FastAPIAppEnvironment
from flyte.models import ActionPhase


# ---------------------------------------------------------------------------
# Web image — only this is built when deploying the frontend app
# ---------------------------------------------------------------------------

_web_image = (
    flyte.Image.from_debian_base(name="automl-webapp")
    .with_apt_packages("git")
    .with_pip_packages(
        "fastapi>=0.115.0",
        "uvicorn>=0.30.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.9",
        "flyte>=2.0.0b52",
        "anthropic>=0.40.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
    )
    .with_source_folder(Path(__file__).parent, copy_contents_only=True)
)


# ---------------------------------------------------------------------------
# Webhook API key — protects /run from unauthorized submissions.
# Set via: union create secret --name automl-webhook-key
# ---------------------------------------------------------------------------

WEBHOOK_API_KEY = os.environ.get("WEBHOOK_API_KEY", "")
_security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(_security),
) -> None:
    if not WEBHOOK_API_KEY:
        return
    if credentials is None or credentials.credentials != WEBHOOK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

class _SuppressStatusPolling(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/status/" not in record.getMessage()


@asynccontextmanager
async def lifespan(application: fastapi.FastAPI):
    logging.getLogger("uvicorn.access").addFilter(_SuppressStatusPolling())
    from flyte._initialize import is_initialized

    if is_initialized():
        # In-cluster the serve runtime has already initialized flyte, including
        # the org (only available via its --org flag). Re-initializing with
        # init_in_cluster() would drop the org and break run submission.
        print("Flyte already initialized by serve runtime", flush=True)
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        print("Flyte initialized from local config", flush=True)
    yield


app = fastapi.FastAPI(title="AutoTrain", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_DEFAULTS = {
    "dataset_link": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "target_column": "Survived",
    "domain": "auto",
    "github_repo": "unionai-oss/autoresearch-experiments",
    "max_experiments": 20,
    "time_budget": 1800,
    "max_samples": "",
}

# The app is stateless: each submission becomes a run named automl-<job_id>,
# labeled so the app's runs are identifiable in the UI/API. Status is always
# looked up from the cluster, so it survives app restarts and redeploys.
_APP_LABEL = {"app": "automl-webapp"}


def _run_name(job_id: str) -> str:
    return f"automl-{job_id}"


# Strong refs to in-flight submission tasks (create_task results are weakly held).
_SUBMISSIONS: set[asyncio.Task] = set()

# Best-effort surfacing of submission failures. A failed submission never
# creates a run, so there is no cluster state to report — this is transient
# diagnostics, not tracking state (lost on restart, which is acceptable).
_SUBMIT_ERRORS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# App environment
# ---------------------------------------------------------------------------

automl_webapp = FastAPIAppEnvironment(
    name="automl-webapp",
    app=app,
    image=_web_image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    port=8080,
    requires_auth=False,
    env_vars=({"WEBHOOK_API_KEY": WEBHOOK_API_KEY} if WEBHOOK_API_KEY else {}),
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"defaults": _DEFAULTS})


@app.post("/run")
async def start_run(
    dataset_link: str      = Form(...),
    target_column: str     = Form(...),
    github_repo: str       = Form(...),
    domain: str            = Form("auto"),
    max_experiments: int   = Form(20),
    time_budget: float     = Form(1800.0),
    max_samples_raw: Optional[str] = Form(None),
    _: None = Security(verify_token),
):
    max_samples = int(max_samples_raw) if max_samples_raw and max_samples_raw.strip() else 0
    # Run names are limited to 30 chars; "automl-" + 20 hex chars fits.
    job_id = uuid.uuid4().hex[:20]

    from pipeline import automl_pipeline  # lazy — keeps `flyte serve` from building task images

    async def _submit():
        # Runs on the same event loop where flyte was initialized — flyte.run.aio
        # must not be driven from a separate thread/event loop.
        try:
            run = await flyte.with_runcontext(
                name=_run_name(job_id),
                labels={**_APP_LABEL, "automl-job-id": job_id},
            ).run.aio(
                automl_pipeline,
                dataset_link=dataset_link,
                target_column=target_column,
                domain=domain,
                github_repo=github_repo,
                max_experiments=max_experiments,
                time_budget_per_experiment_seconds=time_budget,
                max_samples=max_samples,
            )
            print(f"Submitted run {run.name}: {run.url}", flush=True)
        except Exception:
            import traceback
            _SUBMIT_ERRORS[job_id] = traceback.format_exc()
            print(f"Run submission failed for job {job_id}:\n{_SUBMIT_ERRORS[job_id]}", flush=True)

    task = asyncio.create_task(_submit())
    _SUBMISSIONS.add(task)
    task.add_done_callback(_SUBMISSIONS.discard)
    return RedirectResponse(url=f"/status/{job_id}", status_code=303)


@app.get("/status/{job_id}")
async def status_page(job_id: str, request: Request):
    return templates.TemplateResponse(request, "status.html", {"job_id": job_id})


async def _job_status(job_id: str) -> dict:
    """Resolve job status from the cluster — the run itself is the source of truth."""
    try:
        run = await flyte.remote.Run.get.aio(_run_name(job_id))
    except Exception:
        # Run not created yet: submission still in flight (task images may
        # still be building), unless we recorded a submission failure.
        if job_id in _SUBMIT_ERRORS:
            return {"status": "error", "run_url": None, "result": None, "error": _SUBMIT_ERRORS[job_id]}
        return {"status": "starting", "run_url": None, "result": None, "error": None}

    status = {"status": "running", "run_url": run.url, "result": None, "error": None}
    phase = run.phase
    if phase == ActionPhase.SUCCEEDED:
        outputs = await run.outputs.aio()
        status["status"] = "done"
        status["result"] = str(outputs[0]) if len(outputs) else ""
    elif phase in (ActionPhase.FAILED, ActionPhase.ABORTED, ActionPhase.TIMED_OUT):
        error = f"Run {phase.value}"
        try:
            details = await run.details.aio()
            if details.action_details.error_info is not None:
                error = details.action_details.error_info.message
        except Exception:
            pass
        status["status"] = "error"
        status["error"] = error
    return status


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    return JSONResponse(await _job_status(job_id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description="AutoTrain — deploy frontend to Union or run local dev server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--local", action="store_true", help="Run local uvicorn dev server")
    args = parser.parse_args()

    if args.local:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
    else:
        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
        deployed = flyte.serve(automl_webapp)
        print(f"\nApp deployed at: {deployed.url}")
