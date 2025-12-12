"""Complete example showing various environment settings."""

import flyte
import flyte.app

# {{docs-fragment complete-example}}
# Create a custom image
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "python-multipart==0.0.6",
)

# Configure app with various settings
app_env = flyte.app.AppEnvironment(
    name="my-api",
    type="FastAPI",
    image=image,
    port=8080,
    resources=flyte.Resources(
        cpu="2",
        memory="4Gi",
    ),
    secrets=flyte.Secret(key="my-api-key", as_env_var="API_KEY"),
    env_vars={
        "LOG_LEVEL": "INFO",
        "ENVIRONMENT": "production",
    },
    requires_auth=False,  # Public API
    cluster_pool="production-pool",
    description="My production FastAPI service",
)
# {{/docs-fragment complete-example}}

