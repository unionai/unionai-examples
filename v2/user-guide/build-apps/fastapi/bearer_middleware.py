# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""FastAPI app with Bearer token middleware authentication."""

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Get API key from environment variable (loaded from Flyte secret)
# The secret must be created using: flyte create secret API_KEY <your-api-key-value>
API_KEY = os.getenv("API_KEY")

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate requests using Bearer tokens."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        public_paths = ["/docs", "/redoc", "/openapi.json", "/health"]
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Extract Bearer token
        authorization = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return Response(
                content='{"detail": "Missing or invalid Authorization header"}',
                status_code=403,
                media_type="application/json",
            )
        
        token = authorization.split(" ")[1]
        if not API_KEY or token != API_KEY:
            return Response(
                content='{"detail": "Invalid token"}',
                status_code=403,
                media_type="application/json",
            )
        
        return await call_next(request)

app = FastAPI(title="Bearer Middleware Auth API")
app.add_middleware(BearerAuthMiddleware)

@app.get("/health")
async def health():
    """Public health check endpoint."""
    return {"status": "healthy"}

@app.get("/protected")
async def protected():
    """Protected endpoint - requires Bearer token."""
    return {"message": "This endpoint requires Bearer token authentication"}

env = FastAPIAppEnvironment(
    name="bearer-middleware-api",
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

