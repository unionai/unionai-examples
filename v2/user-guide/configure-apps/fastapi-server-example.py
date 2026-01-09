# /// script
# requires-python = "==3.13"
# dependencies = [
#    "fastapi",
#    "uvicorn",
#    "flyte==2.0.0b47",
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
async def server():
    print("Starting server...")
    await uvicorn.Server(uvicorn.Config(app, port=8080)).serve()


@app.get("/")
async def root() -> dict:
    return {"message": "Hello from FastAPI!"}
# {{/docs-fragment fastapi-app}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)
    deployed_app = flyte.serve(env)
    print(f"App served at: {deployed_app.url}")
# {{/docs-fragment deploy}}
