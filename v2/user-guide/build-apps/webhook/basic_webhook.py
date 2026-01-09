# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""A webhook that triggers Flyte tasks."""

import pathlib
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status
import os
from contextlib import asynccontextmanager
import flyte
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment auth}}
WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY", "test-api-key")
security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    """Verify the API key from the bearer token."""
    if credentials.credentials != WEBHOOK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials
# {{/docs-fragment auth}}

# {{docs-fragment lifespan}}
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte before accepting requests."""
    await flyte.init_in_cluster.aio()
    yield
    # Cleanup if needed
# {{/docs-fragment lifespan}}

# {{docs-fragment app}}
app = FastAPI(
    title="Flyte Webhook Runner",
    description="A webhook service that triggers Flyte task runs",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
# {{/docs-fragment app}}

# {{docs-fragment webhook-endpoint}}
@app.post("/run-task/{project}/{domain}/{name}/{version}")
async def run_task(
    project: str,
    domain: str,
    name: str,
    version: str,
    inputs: dict,
    credentials: HTTPAuthorizationCredentials = Security(verify_token),
):
    """
    Trigger a Flyte task run via webhook.
    
    Returns information about the launched run.
    """
    # Fetch the task
    task = remote.Task.get(
        project=project,
        domain=domain,
        name=name,
        version=version,
    )
    
    # Run the task
    run = await flyte.run.aio(task, **inputs)
    
    return {
        "url": run.url,
        "id": run.id,
        "status": "started",
    }
# {{/docs-fragment webhook-endpoint}}

# {{docs-fragment env}}
env = FastAPIAppEnvironment(
    name="webhook-runner",
    app=app,
    description="A webhook service that triggers Flyte task runs",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,  # We handle auth in the app
    env_vars={"WEBHOOK_API_KEY": os.getenv("WEBHOOK_API_KEY", "test-api-key")},
)
# {{/docs-fragment env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed webhook: {app_deployment[0].summary_repr()}")
# {{/docs-fragment deploy}}

