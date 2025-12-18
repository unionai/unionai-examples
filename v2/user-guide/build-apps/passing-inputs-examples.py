"""Examples showing different ways to pass inputs into apps."""

import flyte
import flyte.app
import flyte.io

# {{docs-fragment basic-input-types}}
# String inputs
app_env = flyte.app.AppEnvironment(
    name="configurable-app",
    inputs=[
        flyte.app.Input(name="environment", value="production"),
        flyte.app.Input(name="log_level", value="INFO"),
    ],
    # ...
)

# File inputs
app_env2 = flyte.app.AppEnvironment(
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

# Directory inputs
app_env3 = flyte.app.AppEnvironment(
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
# {{/docs-fragment basic-input-types}}

# {{docs-fragment runoutput-example}}
# Delayed inputs with RunOutput
env = flyte.TaskEnvironment(name="training-env")

@env.task
async def train_model() -> flyte.io.File:
    # ... training logic ...
    return await flyte.io.File.from_local("/tmp/trained-model.pkl")

# Use the task output as an app input
app_env4 = flyte.app.AppEnvironment(
    name="serving-app",
    inputs=[
        flyte.app.Input(
            name="model",
            value=flyte.app.RunOutput(type="file", run_name="training_run", task_name="train_model"),
            mount="/app/model",
        ),
    ],
    # ...
)
# {{/docs-fragment runoutput-example}}

# {{docs-fragment appendpoint-example}}
# Delayed inputs with AppEndpoint
app1_env = flyte.app.AppEnvironment(name="backend-api")

app2_env = flyte.app.AppEnvironment(
    name="frontend-app",
    inputs=[
        flyte.app.Input(
            name="backend_url",
            value=flyte.app.AppEndpoint(app_name="backend-api"),
            env_var="BACKEND_URL",  # app1_env's endpoint will be available as an environment variable
        ),
    ],
    # ...
)
# {{/docs-fragment appendpoint-example}}

# {{docs-fragment runoutput-serving-example}}
# Example: Using RunOutput for model serving
import joblib
from sklearn.ensemble import RandomForestClassifier
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI

# Training task
training_env = flyte.TaskEnvironment(name="training-env")

@training_env.task
async def train_model_task() -> flyte.io.File:
    """Train a model and return it."""

    model = RandomForestClassifier()

    # ... training logic ...

    path = "./trained-model.pkl"
    joblib.dump(model, path)
    return await flyte.io.File.from_local(path)

# Serving app that uses the trained model
app = FastAPI()
serving_env = FastAPIAppEnvironment(
    name="model-serving-app",
    app=app,
    inputs=[
        flyte.app.Input(
            name="model",
            value=flyte.app.RunOutput(
                type="file",
                task_name="training-env.train_model_task"
            ),
            mount="/app/model",
            env_var="MODEL_PATH",
        ),
    ],
)
# {{/docs-fragment runoutput-serving-example}}

