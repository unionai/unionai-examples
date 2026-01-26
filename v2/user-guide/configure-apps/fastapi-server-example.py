# /// script
# requires-python = "==3.13"
# dependencies = [
#    "fastapi",
#    "uvicorn",
#    "flyte==2.0.0b49",
# ]
# ///

import fastapi
import uvicorn

import flyte
from flyte.app.extras import FastAPIAppEnvironment

# {{docs-fragment fastapi-app}}
app = fastapi.FastAPI()

env = FastAPIAppEnvironment(
    name="configure-fastapi-example",
    app=app,
    image=flyte.Image.from_uv_script(__file__, name="configure-fastapi-example"),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
    port=8080,
)

@env.server
def server():
    print("Starting server...")
    uvicorn.run(app, port=8080)


@app.get("/")
async def root() -> dict:
    return {"message": "Hello from FastAPI!"}
# {{/docs-fragment fastapi-app}}


# {{docs-fragment on-startup-decorator}}
state = {}

@env.on_startup
async def app_startup():
    print("App started up")
    state["data"] = ["Here's", "some", "data"]
# {{/docs-fragment on-startup-decorator}}

# {{docs-fragment on-shutdown-decorator}}
@env.on_shutdown
async def app_shutdown():
    print("App shut down")
    state.clear()  # clears the data
# {{/docs-fragment on-shutdown-decorator}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    deployed_app = flyte.serve(env)
    print(f"App served at: {deployed_app.url}")
# {{/docs-fragment deploy}}
