# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
# ]
# ///

"""Multi-file FastAPI app example."""

from fastapi import FastAPI
from module import function  # Import from another file
import pathlib

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment app-definition}}
app = FastAPI(title="Multi-file FastAPI Demo")

app_env = FastAPIAppEnvironment(
    name="fastapi-multi-file",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    # FastAPIAppEnvironment automatically includes necessary files
    # But you can also specify explicitly:
    # include=["app.py", "module.py"],
)
# {{/docs-fragment app-definition}}

# {{docs-fragment endpoint}}
@app.get("/")
async def root():
    return function()  # Uses function from module.py
# {{/docs-fragment endpoint}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(app_env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")
# {{/docs-fragment deploy}}

