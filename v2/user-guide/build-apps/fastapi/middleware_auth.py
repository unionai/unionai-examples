# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "fastapi",
# ]
# ///

"""FastAPI app with middleware-based authentication."""

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette import status
import os
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Get API key from environment variable (loaded from Flyte secret)
# The secret must be created using: flyte create secret API_KEY <your-api-key-value>
API_KEY = os.getenv("API_KEY")

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to authenticate all requests except public endpoints."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health"]:
            return await call_next(request)
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_KEY:
            return Response(
                content='{"detail": "Invalid or missing API key"}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json",
            )
        
        return await call_next(request)

app = FastAPI(title="Middleware Authenticated API")
app.add_middleware(AuthMiddleware)

@app.get("/health")
async def health():
    """Public health check endpoint."""
    return {"status": "healthy"}

@app.get("/protected")
async def protected():
    """Protected endpoint - requires X-API-Key header."""
    return {"message": "This endpoint requires authentication"}

env = FastAPIAppEnvironment(
    name="middleware-auth-api",
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

