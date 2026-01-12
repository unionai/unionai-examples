# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""A webhook with Pydantic validation."""

import pathlib
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status
import os
from contextlib import asynccontextmanager
from pydantic import BaseModel
import flyte
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte before accepting requests."""
    await flyte.init_in_cluster.aio()
    yield


app = FastAPI(
    title="Flyte Webhook Runner with Validation",
    description="A webhook service that triggers Flyte task runs with Pydantic validation",
    version="1.0.0",
    lifespan=lifespan,
)


# {{docs-fragment validation-model}}
class TaskInput(BaseModel):
    data: dict
    priority: int = 0
# {{/docs-fragment validation-model}}


# {{docs-fragment validated-webhook}}
@app.post("/run-task/{project}/{domain}/{name}/{version}")
async def run_task(
    project: str,
    domain: str,
    name: str,
    version: str,
    inputs: TaskInput,  # Validated input
    credentials: HTTPAuthorizationCredentials = Security(verify_token),
):
    task = remote.Task.get(
        project=project,
        domain=domain,
        name=name,
        version=version,
    )
    
    run = await flyte.run.aio(task, **inputs.model_dump())
    
    return {
        "run_id": run.id,
        "url": run.url,
    }
# {{/docs-fragment validated-webhook}}


env = FastAPIAppEnvironment(
    name="webhook-with-validation",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    env_vars={"WEBHOOK_API_KEY": os.getenv("WEBHOOK_API_KEY", "test-api-key")},
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed webhook: {app_deployment[0].summary_repr()}")
