"""Basic FastAPI authentication using dependency injection."""

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Get API key from environment variable (loaded from Flyte secret)
# The secret must be created using: flyte create secret API_KEY <your-api-key-value>
API_KEY = os.getenv("API_KEY")
security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    """Verify the API key from the bearer token."""
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured",
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials

app = FastAPI(title="Authenticated API")

@app.get("/public")
async def public_endpoint():
    """Public endpoint that doesn't require authentication."""
    return {"message": "This is public"}

@app.get("/protected")
async def protected_endpoint(
    credentials: HTTPAuthorizationCredentials = Security(verify_token),
):
    """Protected endpoint that requires authentication."""
    return {
        "message": "This is protected",
        "user": credentials.credentials,
    }

env = FastAPIAppEnvironment(
    name="authenticated-api",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,  # We handle auth in the app
    secrets=flyte.Secret(key="API_KEY", as_env_var="API_KEY"),
)

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app_deployment = flyte.deploy(env)
    print(f"API URL: {app_deployment[0].url}")
    print(f"Swagger docs: {app_deployment[0].url}/docs")

