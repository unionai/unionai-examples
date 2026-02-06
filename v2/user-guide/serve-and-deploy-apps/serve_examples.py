# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# ///

"""Serve examples for the how-app-serving-works.md documentation."""

import logging
import flyte
import flyte.app


# {{docs-fragment basic-serve}}
app_env = flyte.app.AppEnvironment(
    name="my-dev-app",
    parameters=[flyte.app.Parameter(name="model_path", value="s3://bucket/models/model.pkl")],
    # ...
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(app_env)
    print(f"App served at: {app.url}")
# {{/docs-fragment basic-serve}}


# {{docs-fragment override-parameters}}
app = flyte.with_servecontext(
    input_values={
        "my-dev-app": {
            "model_path": "s3://bucket/models/test-model.pkl",
        }
    }
).serve(app_env)
# {{/docs-fragment override-parameters}}


# {{docs-fragment advanced-serving}}
app = flyte.with_servecontext(
    version="v1.0.0",
    project="my-project",
    domain="development",
    env_vars={"LOG_LEVEL": "DEBUG"},
    input_values={"app-name": {"input": "value"}},
    cluster_pool="dev-pool",
    log_level=logging.INFO,
    log_format="json",
    dry_run=False,
).serve(app_env)
# {{/docs-fragment advanced-serving}}


# {{docs-fragment return-value}}
app = flyte.serve(app_env)
print(f"URL: {app.url}")
print(f"Endpoint: {app.endpoint}")
print(f"Status: {app.deployment_status}")
# {{/docs-fragment return-value}}
