# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
# ]
# ///

"""FastAPI app with multiple authentication methods."""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from starlette import status
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Get API keys from environment variables (loaded from Flyte secrets)
# The secrets must be created using:
#   flyte create secret API_KEY <your-api-key-value>
#   flyte create secret BEARER_TOKEN <your-bearer-token-value>
API_KEY = os.getenv("API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Bearer token authentication
bearer_scheme = HTTPBearer()

async def verify_bearer(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
):
    """Verify Bearer token."""
    if not BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="BEARER_TOKEN not configured")
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    return credentials

# API key header authentication
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key from header."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

app = FastAPI(title="Multi-Auth API")

@app.get("/bearer-protected")
async def bearer_endpoint(credentials = Depends(verify_bearer)):
    """Endpoint protected with Bearer token."""
    return {"message": "Bearer authenticated"}

@app.get("/apikey-protected")
async def apikey_endpoint(api_key = Depends(verify_api_key)):
    """Endpoint protected with API key header."""
    return {"message": "API key authenticated"}

env = FastAPIAppEnvironment(
    name="multiple-auth-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    secrets=[
        flyte.Secret(key="API_KEY", as_env_var="API_KEY"),
        flyte.Secret(key="BEARER_TOKEN", as_env_var="BEARER_TOKEN"),
    ],
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"Deployed: {app_deployment[0].summary_repr()}")

