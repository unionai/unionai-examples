# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
#    "websockets",
# ]
# ///

"""A WebSocket app that triggers Flyte tasks and streams updates."""

import json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import flyte
import flyte.remote as remote
from flyte.app.extras import FastAPIAppEnvironment


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Flyte before accepting requests."""
    await flyte.init_in_cluster.aio()
    yield


app = FastAPI(
    title="WebSocket Task Runner",
    description="Triggers Flyte tasks via WebSocket and streams updates",
    version="1.0.0",
    lifespan=lifespan,
)


# {{docs-fragment task-runner-websocket}}
@app.websocket("/task-runner")
async def task_runner(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive task request
            message = await websocket.receive_text()
            request = json.loads(message)

            # Trigger Flyte task
            task = remote.Task.get(
                project=request["project"],
                domain=request["domain"],
                name=request["task"],
                version=request["version"],
            )

            run = await flyte.run.aio(task, **request["inputs"])

            # Send run info back
            await websocket.send_json({
                "run_id": run.id,
                "url": run.url,
                "status": "started",
            })

            # Optionally stream updates
            async for update in run.stream():
                await websocket.send_json({
                    "status": update.status,
                    "message": update.message,
                })

    except WebSocketDisconnect:
        pass
# {{/docs-fragment task-runner-websocket}}


env = FastAPIAppEnvironment(
    name="task-runner-websocket",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "websockets",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed WebSocket task runner: {app_deployment[0].summary_repr()}")
