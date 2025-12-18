"""Examples showing different app input types."""

import flyte
import flyte.app
import flyte.io

# {{docs-fragment basic-inputs}}
# Basic input syntax
app_env = flyte.app.AppEnvironment(
    name="my-app",
    inputs=[
        flyte.app.Input(name="model_path", value="s3://bucket/models/model.pkl"),
        flyte.app.Input(name="api_key", value="my-secret-key"),
    ],
    # ...
)
# {{/docs-fragment basic-inputs}}

# {{docs-fragment string-inputs}}
# String inputs
app_env2 = flyte.app.AppEnvironment(
    name="configurable-app",
    inputs=[
        flyte.app.Input(name="environment", value="production"),
        flyte.app.Input(name="log_level", value="INFO"),
    ],
    # ...
)
# {{/docs-fragment string-inputs}}

# {{docs-fragment file-inputs}}
# File inputs
app_env3 = flyte.app.AppEnvironment(
    name="app-with-model",
    inputs=[
        flyte.app.Input(
            name="model_file",
            value=flyte.io.File("s3://bucket/models/model.pkl"),
            mount="/app/models",
        ),
    ],
    # ...
)
# {{/docs-fragment file-inputs}}

# {{docs-fragment directory-inputs}}
# Directory inputs
app_env4 = flyte.app.AppEnvironment(
    name="app-with-data",
    inputs=[
        flyte.app.Input(
            name="data_dir",
            value=flyte.io.Dir("s3://bucket/data/"),
            mount="/app/data",
        ),
    ],
    # ...
)
# {{/docs-fragment directory-inputs}}

# {{docs-fragment runoutput-input}}
# Delayed inputs with RunOutput
env = flyte.TaskEnvironment(name="training-env")

@env.task
async def train_model() -> flyte.io.File:
    # ... training logic ...
    return await flyte.io.File.from_local("/tmp/trained-model.pkl")

# Use the task output as an app input
app_env5 = flyte.app.AppEnvironment(
    name="serving-app",
    inputs=[
        flyte.app.Input(
            name="model",
            value=flyte.app.RunOutput(run_name="training_run", task_name="train_model"),
            mount="/app/model",
        ),
    ],
    # ...
)
# {{/docs-fragment runoutput-input}}

# {{docs-fragment appendpoint-input}}
# Delayed inputs with AppEndpoint
app1_env = flyte.app.AppEnvironment(name="backend-api", ...)

app2_env = flyte.app.AppEnvironment(
    name="frontend-app",
    inputs=[
        flyte.app.Input(
            name="backend_url",
            value=flyte.app.AppEndpoint(app_name="backend-api"),
        ),
    ],
    # ...
)
# {{/docs-fragment appendpoint-input}}

# {{docs-fragment override-inputs}}
# Overriding inputs at serve/deploy time
# Override inputs when serving
app = flyte.serve(
    app_env,
    input_values={
        "my-app": {
            "model_path": "s3://bucket/new-model.pkl",
            "api_key": "new-key",
        }
    },
)

# Override inputs when deploying
app = flyte.deploy(
    app_env,
    input_values={
        "my-app": {
            "model_path": "s3://bucket/new-model.pkl",
        }
    },
)
# {{/docs-fragment override-inputs}}

