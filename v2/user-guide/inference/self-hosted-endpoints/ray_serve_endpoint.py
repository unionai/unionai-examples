# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "ray[serve]",
#    "torch",
#    "transformers",
# ]
# ///

"""Serve a Hugging Face model with Ray Serve on a generic AppEnvironment."""

import sys
import time
from pathlib import Path

import flyte
import flyte.app

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
PORT = 8080

file_name = Path(__file__).name

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="ray-serve-endpoint",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "ray[serve]", "torch", "transformers"
    ),
    args=["python", file_name, "--server"],
    port=PORT,
    resources=flyte.Resources(cpu="4", memory="8Gi", disk="10Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)
# {{/docs-fragment app-env}}


# {{docs-fragment server}}
def serve() -> None:
    from ray import serve as ray_serve
    from starlette.requests import Request
    from transformers import pipeline

    @ray_serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
    class Classifier:
        def __init__(self):
            self.classifier = pipeline("text-classification", model=MODEL)

        async def __call__(self, request: Request) -> dict:
            body = await request.json()
            return self.classifier(body["text"])[0]

    # Bind the Ray Serve proxy to the app port on all interfaces.
    ray_serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    ray_serve.run(Classifier.bind(), route_prefix="/predict")

    # `serve.run` returns once the deployment is ready. Keep the container alive.
    while True:
        time.sleep(60)
# {{/docs-fragment server}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    if "--server" in sys.argv:
        serve()
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        deployment = flyte.serve(app_env)
        print(f"Deployed: {deployment.url}")
# {{/docs-fragment deploy}}
