# /// script
# requires-python = "==3.13"
# dependencies = [
#    "fastapi",
#    "flyte>=2.0.0b52",
# ]
# ///

import flyte
import flyte.app

# {{docs-fragment complete-example}}
# Configure app with various settings
app_env = flyte.app.AppEnvironment(
    name="my-api",
    type="FastAPI",
    image=flyte.Image.from_uv_script(__file__, name="environment-settings-example"),
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

