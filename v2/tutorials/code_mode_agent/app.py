"""Code Mode analytics agent, served as a Flyte app that launches durable runs.

Chat with a dataset: you ask a question, and the app launches a Flyte **run** to
answer it. Inside that run, Claude writes a Python program, it executes in the
Monty sandbox, and every ``query`` the program makes is dispatched as a durable,
observable child task. The app is a thin front end; the analysis is a real
workflow you can click into in the Union UI.

Why a run and not a direct call: an app's request handler has no task context, so
calling a task directly runs it locally in the app. To get durable execution the
app launches the work with ``flyte.run`` — inside that run there *is* a task
context, so the sandbox's tool calls dispatch as tasks. This is the "call task
from app" pattern from the hybrid app-task graphs guide: initialize the client
once with ``flyte.init_in_cluster`` in the app lifespan, then ``flyte.run.aio(...)``
per request.

``init_in_cluster`` resolves project, domain, endpoint, and identity from the app
pod automatically. The one thing app pods aren't given is the org, so we read it
from the config at deploy and pass it in via the ANALYSIS_ORG env var.

The durable half (the ``analyze`` task, the ``query`` task, the agent) lives in
``analysis.py`` so it can run in a task image without the web dependencies.

Install dependencies::

    pip install 'flyte[sandbox]' anthropic duckdb pandas

Run::

    flyte create secret sam_anthropic_api_key <your-anthropic-key>
    python app.py
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

import tools
from analysis import ChatResponse, analyze, env as agent_env, tool_descriptions
from ui import CHAT_HTML


# {{docs-fragment lifespan}}
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await flyte.init_in_cluster.aio(org=os.getenv("ANALYSIS_ORG") or None)
    # Build the data sample once now (it's cheap, pure Python) so the first "Preview the
    # dataset" click is instant rather than paying for it on the request.
    tools.dataset_sample()
    yield
# {{/docs-fragment lifespan}}


app = FastAPI(title="Code Mode Analytics Agent", lifespan=lifespan)

# {{docs-fragment app_env}}
env = FastAPIAppEnvironment(
    name="code-mode-analytics",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    scaling=flyte.app.Scaling(replicas=1),
    depends_on=[agent_env],
    # Every request launches a run (compute + a paid LLM call), so gate the app
    # behind Union auth.
    requires_auth=True,
)
# {{/docs-fragment app_env}}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/tools")
async def get_tools() -> list[dict]:
    """Tool metadata for the UI sidebar (auto-generated from the registry)."""
    return tool_descriptions()


# {{docs-fragment dataset}}
@app.get("/api/dataset")
async def get_dataset() -> dict:
    """A cheap in-process peek at the data: the schema and a sample of rows.

    This does *not* launch a run or touch the query engine — it reads a few row dicts
    straight from the (cached) generator, so it returns instantly. The durable analysis
    path is still ``/api/chat``.
    """
    headers, rows, count = tools.dataset_sample(12)
    table = await tools.create_table(
        f"Sample of orders ({len(rows)} of {count:,} rows)", headers, rows
    )
    summary = (
        f"The `orders` table — ecommerce orders for 2024, about {count:,} rows. "
        "Columns: " + ", ".join(headers) + ". Ask a question to analyze it."
    )
    return {"summary": summary, "table": table, "count": count}
# {{/docs-fragment dataset}}


# {{docs-fragment chat}}
@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Launch a durable analysis run, wait for it, and return charts + summary + the run link."""
    try:
        run = await flyte.run.aio(analyze, message=req.message)
        await run.wait.aio()
        result: ChatResponse = (await run.outputs.aio())[0]
        result.run_url = run.url
        return result
    except Exception as exc:  # noqa: BLE001 — surface any launch/run failure to the UI
        return ChatResponse(error=str(exc))
# {{/docs-fragment chat}}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=CHAT_HTML)


# {{docs-fragment deploy}}
if __name__ == "__main__":
    # Remote image builder so no local Docker is needed to build the app + task images.
    flyte.init_from_config(image_builder="remote")

    # App pods don't inherit the org, but launching a run needs it. Read it from the
    # deploy config (or the ANALYSIS_ORG override) and inject it as an env var the app's
    # lifespan passes to init_in_cluster(org=...).
    from flyte.config import Config

    org = os.getenv("ANALYSIS_ORG") or (Config.auto().task.org or "")
    if org:
        env.env_vars = {**(env.env_vars or {}), "ANALYSIS_ORG": org}
    else:
        print("WARNING: no org found; set ANALYSIS_ORG=<your-org> and redeploy.")

    handle = flyte.serve(env)
    print(f"Deployed Code Mode Analytics Agent: {handle.url}")
# {{/docs-fragment deploy}}
