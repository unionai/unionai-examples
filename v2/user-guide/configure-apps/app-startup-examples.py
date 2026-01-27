# /// script
# requires-python = "==3.13"
# dependencies = [
#    "fastapi",
#    "flyte>=2.0.0b52",
# ]
# ///

import flyte
import flyte.app

# {{docs-fragment args-with-default-command}}
# Using args with default command
app_env = flyte.app.AppEnvironment(
    name="streamlit-app",
    args="streamlit run main.py --server.port 8080",
    port=8080,
    include=["main.py"],
    # command is None, so default Flyte command is used
)
# {{/docs-fragment args-with-default-command}}

# {{docs-fragment explicit-command}}
# Using explicit command
app_env2 = flyte.app.AppEnvironment(
    name="streamlit-hello",
    command="streamlit hello --server.port 8080",
    port=8080,
    # No args needed since command includes everything
)
# {{/docs-fragment explicit-command}}

# {{docs-fragment command-with-args}}
# Using command with args
app_env3 = flyte.app.AppEnvironment(
    name="custom-app",
    command="python -m myapp",
    args="--option1 value1 --option2 value2",
    # This runs: python -m myapp --option1 value1 --option2 value2
)
# {{/docs-fragment command-with-args}}

# {{docs-fragment fastapi-auto-command}}
# FastAPIAppEnvironment automatically sets command
from flyte.app.extras import FastAPIAppEnvironment
from fastapi import FastAPI

app = FastAPI()

env = FastAPIAppEnvironment(
    name="my-api",
    app=app,
    # You typically don't need to specify command or args, since the
    # FastAPIAppEnvironment automatically uses the bundled code to serve the
    # app via uvicorn.
)
# {{/docs-fragment fastapi-auto-command}}

