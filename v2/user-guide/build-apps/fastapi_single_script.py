"""A single-script FastAPI app example - the simplest FastAPI app."""

from fastapi import FastAPI
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment fastapi-app}}
app = FastAPI(
    title="Simple FastAPI App",
    description="A minimal single-script FastAPI application",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
# {{/docs-fragment fastapi-app}}

# {{docs-fragment app-env}}
app_env = FastAPIAppEnvironment(
    name="fastapi-single-script",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.serve(app_env)
    print(f"Deployed: {app_deployment[0].url}")
    print(f"API docs: {app_deployment[0].url}/docs")
# {{/docs-fragment deploy}}

