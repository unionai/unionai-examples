# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""A basic FastAPI app example."""

from fastapi import FastAPI
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment fastapi-app}}
app = FastAPI(
    title="My API",
    description="A simple FastAPI application",
    version="1.0.0",
)
# {{/docs-fragment fastapi-app}}

# {{docs-fragment fastapi-env}}
env = FastAPIAppEnvironment(
    name="my-fastapi-app",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)
# {{/docs-fragment fastapi-env}}

# {{docs-fragment endpoints}}
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
# {{/docs-fragment endpoints}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")
# {{/docs-fragment deploy}}

