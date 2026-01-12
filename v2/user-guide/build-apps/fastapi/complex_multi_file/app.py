# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""Complex multi-file FastAPI app example."""

from pathlib import Path
from fastapi import FastAPI
from models.user import User
from services.auth import authenticate
from utils.helpers import format_response

import flyte
from flyte.app.extras import FastAPIAppEnvironment


# {{docs-fragment complex-app}}
app = FastAPI(title="Complex Multi-file App")


@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = User(id=user_id, name="John Doe")
    return format_response(user)
# {{/docs-fragment complex-app}}


# {{docs-fragment complex-env}}
app_env = FastAPIAppEnvironment(
    name="complex-app",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
        "pydantic",
    ),
    # Include all necessary files
    include=[
        "app.py",
        "models/",
        "services/",
        "utils/",
    ],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)
# {{/docs-fragment complex-env}}


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app_deployment = flyte.deploy(app_env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")
