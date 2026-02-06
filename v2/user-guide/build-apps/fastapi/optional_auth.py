# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
# ]
# ///

"""FastAPI app with optional authentication."""

from typing import Optional
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Get API key from environment variable (loaded from Flyte secret)
# The secret must be created using: flyte create secret API_KEY <your-api-key-value>
API_KEY = os.getenv("API_KEY")

async def verify_optional_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False)),
):
    """Verify token if provided, but allow anonymous access."""
    if credentials is None:
        return None
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
    return credentials

app = FastAPI(title="Optional Auth API")

@app.get("/optional-auth")
async def optional_auth_endpoint(credentials = Security(verify_optional_token)):
    """Endpoint that works with or without authentication."""
    if credentials:
        return {"message": "Authenticated user", "token": credentials.credentials}
    return {"message": "Anonymous user"}

env = FastAPIAppEnvironment(
    name="optional-auth-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    secrets=flyte.Secret(key="API_KEY", as_env_var="API_KEY"),
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")

