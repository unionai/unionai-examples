# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b45",
#    "streamlit",
# ]
# ///

import flyte
import flyte.app
import flyte.io

# {{docs-fragment basic-inputs}}
# Basic parameter syntax
app_env = flyte.app.AppEnvironment(
    name="my-app",
    parameters=[
        flyte.app.Parameter(name="key", value="value"),
        flyte.app.Parameter(
            name="model_path",
            value=flyte.io.File.from_existing_remote("s3://bucket/models/model.pkl"),
        ),
    ],
    image=flyte.Image.from_uv_script(__file__, name="app-parameters-examples"),
    args=["streamlit", "hello", "--server.port", "8080"],
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    port=8080,
)
# {{/docs-fragment basic-inputs}}

# {{docs-fragment string-inputs}}
# String parameters
app_env2 = flyte.app.AppEnvironment(
    name="configurable-app",
    parameters=[
        flyte.app.Parameter(name="environment", value="production"),
        flyte.app.Parameter(name="log_level", value="INFO"),
    ],
)
# {{/docs-fragment string-inputs}}

# {{docs-fragment file-inputs}}
# File parameters
app_env3 = flyte.app.AppEnvironment(
    name="app-with-model",
    parameters=[
        flyte.app.Parameter(
            name="model_file",
            value=flyte.io.File.from_existing_remote("s3://bucket/models/model.pkl"),
            mount="/app/models",
        ),
    ],
)
# {{/docs-fragment file-inputs}}

# {{docs-fragment directory-inputs}}
# Directory parameters
app_env4 = flyte.app.AppEnvironment(
    name="app-with-data",
    parameters=[
        flyte.app.Parameter(
            name="data_dir",
            value=flyte.io.Dir.from_existing_remote("s3://bucket/data/"),
            mount="/app/data",
        ),
    ],
)
# {{/docs-fragment directory-inputs}}

# {{docs-fragment runoutput-input}}
# Delayed parameters with RunOutput
env = flyte.TaskEnvironment(name="training-env")

@env.task
async def train_model() -> flyte.io.File:
    # ... training logic ...
    import pickle

    path = "/tmp/trained-model.pkl"
    with open(path, "wb") as f:
        pickle.dump("dummy model", f)
    return await flyte.io.File.from_local(path)

# Use the task output as an app parameter
app_env5 = flyte.app.AppEnvironment(
    name="serving-app",
    parameters=[
        flyte.app.Parameter(
            name="model",
            value=flyte.app.RunOutput(type="file", run_name="abc123", task_name="training-env.train_model"),
            mount="/app/model",
        ),
    ],
)
# {{/docs-fragment runoutput-input}}

# {{docs-fragment appendpoint-input}}
# Delayed parameters with AppEndpoint
app1_env = flyte.app.AppEnvironment(name="backend-api")

app2_env = flyte.app.AppEnvironment(
    name="frontend-app",
    parameters=[
        flyte.app.Parameter(
            name="backend_url",
            value=flyte.app.AppEndpoint(app_name="backend-api"),
        ),
    ],
    # ...
)
# {{/docs-fragment appendpoint-input}}

# {{docs-fragment override-inputs}}
# Overriding parameters at serve/deploy time
# Override parameters when serving
if __name__ == "__main__":
    flyte.init_from_config()

    # run the training task
    run = flyte.run(train_model)
    run.wait()
    model_file = run.outputs()[0]

    # serve the app with overridden parameters
    app = flyte.with_servecontext(
        parameter_values={
            "my-app": {
                "model_path": flyte.io.File.from_existing_remote(model_file.path),
                "api_key": "new-key",
            }
        }).serve(app_env)
# {{/docs-fragment override-inputs}}
