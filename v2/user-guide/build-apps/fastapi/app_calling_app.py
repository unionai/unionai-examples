"""Example of one app calling another app."""

import httpx
from fastapi import FastAPI
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "fastapi", "uvicorn", "httpx"
)

# {{docs-fragment backend-app}}
app1 = FastAPI(
    title="App 1",
    description="A FastAPI app that runs some computations",
)

env1 = FastAPIAppEnvironment(
    name="app1-is-called-by-app2",
    app=app1,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)
# {{/docs-fragment backend-app}}

# {{docs-fragment frontend-app}}
app2 = FastAPI(
    title="App 2",
    description="A FastAPI app that proxies requests to another FastAPI app",
)

env2 = FastAPIAppEnvironment(
    name="app2-calls-app1",
    app=app2,
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    depends_on=[env1],  # Depends on backend-api
)
# {{/docs-fragment frontend-app}}

# {{docs-fragment backend-endpoint}}
@app1.get("/greeting/{name}")
async def greeting(name: str) -> str:
    return f"Hello, {name}!"
# {{/docs-fragment backend-endpoint}}

# {{docs-fragment frontend-endpoints}}
@app2.get("/app1-endpoint")
async def get_app1_endpoint() -> str:
    return env1.endpoint  # Access the backend endpoint

@app2.get("/greeting/{name}")
async def greeting_proxy(name: str):
    """Proxy that calls the backend app."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{env1.endpoint}/greeting/{name}")
        response.raise_for_status()
        return response.json()
# {{/docs-fragment frontend-endpoints}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env2)
    print(f"Deployed FastAPI app: {deployments[0].env_repr()}")
# {{/docs-fragment deploy}}

