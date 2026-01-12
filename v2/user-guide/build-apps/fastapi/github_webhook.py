# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""A GitHub webhook that triggers Flyte tasks based on GitHub events."""

import pathlib
import hmac
import hashlib
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Header, HTTPException
import flyte
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte before accepting requests."""
    await flyte.init_in_cluster.aio()
    yield


app = FastAPI(
    title="GitHub Webhook Handler",
    description="Triggers Flyte tasks based on GitHub events",
    version="1.0.0",
    lifespan=lifespan,
)


# {{docs-fragment github-webhook}}
@app.post("/github-webhook")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
):
    """Handle GitHub webhook events."""
    body = await request.body()
    
    # Verify signature
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    signature = hmac.new(
        secret.encode(),
        body,
        hashlib.sha256
    ).hexdigest()
    
    expected_signature = f"sha256={signature}"
    if not hmac.compare_digest(x_hub_signature_256, expected_signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Process webhook
    event = await request.json()
    event_type = request.headers.get("X-GitHub-Event")
    
    if event_type == "push":
        # Trigger deployment task
        task = remote.Task.get(
            project="my-project",
            domain="development",
            name="deploy-task",
            version="v1",
        )
        run = await flyte.run.aio(task, commit=event["after"])
        return {"run_id": run.id, "url": run.url}
    
    return {"status": "ignored"}
# {{/docs-fragment github-webhook}}


# {{docs-fragment env}}
env = FastAPIAppEnvironment(
    name="github-webhook",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    secrets=flyte.Secret(key="GITHUB_WEBHOOK_SECRET", as_env_var="GITHUB_WEBHOOK_SECRET"),
)
# {{/docs-fragment env}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed GitHub webhook: {app_deployment[0].summary_repr()}")
