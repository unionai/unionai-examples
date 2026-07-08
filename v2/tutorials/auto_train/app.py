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
from concurrent.futures import ThreadPoolExecutor
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
from flyte.app.extras import FastAPIAppEnvironment


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

_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(application: fastapi.FastAPI):
    logging.getLogger("uvicorn.access").addFilter(_SuppressStatusPolling())
    try:
        await flyte.init_in_cluster.aio()
        print("Flyte initialized in cluster", flush=True)
    except Exception:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        print("Flyte initialized from local config", flush=True)
    yield
    _executor.shutdown(wait=False)


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

# job_id -> {status, run_url, result, error}
_JOBS: dict[str, dict] = {}


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
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "starting", "run_url": None, "result": None, "error": None}

    def _submit():
        import asyncio as _aio
        from pipeline import automl_pipeline  # lazy — imported in background thread

        async def _run():
            try:
                run = await flyte.run.aio(
                    automl_pipeline,
                    dataset_link=dataset_link,
                    target_column=target_column,
                    domain=domain,
                    github_repo=github_repo,
                    job_id=job_id,
                    webapp_endpoint=automl_webapp.endpoint or "",
                    max_experiments=max_experiments,
                    time_budget_per_experiment_seconds=time_budget,
                    max_samples=max_samples,
                )
                _JOBS[job_id]["run_url"] = run.url
                _JOBS[job_id]["status"] = "running"
            except Exception as exc:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["error"] = str(exc)

        loop = _aio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    _executor.submit(_submit)
    return RedirectResponse(url=f"/status/{job_id}", status_code=303)


@app.post("/result/{job_id}")
async def receive_result(job_id: str, request: Request):
    """Callback: research task POSTs training result here when done."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = await request.json()
    _JOBS[job_id]["status"] = "done"
    _JOBS[job_id]["result"] = payload.get("result", "")
    return {"ok": True}


@app.get("/status/{job_id}")
async def status_page(job_id: str, request: Request):
    job = _JOBS.get(job_id, {"status": "not_found"})
    return templates.TemplateResponse(request, "status.html", {"job_id": job_id, "job": job})


@app.get("/api/status/{job_id}")
async def api_status(job_id: str):
    return JSONResponse(_JOBS.get(job_id, {"status": "not_found"}))


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
