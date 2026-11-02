# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "bentoml",
#    "torch",
#    "transformers",
# ]
# ///

"""Serve a Hugging Face model with BentoML on a generic AppEnvironment."""

from pathlib import Path

import bentoml
import flyte
import flyte.app

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
PORT = 8080


# {{docs-fragment service}}
@bentoml.service(workers=1)
class Classifier:
    def __init__(self):
        from transformers import pipeline

        self.classifier = pipeline("text-classification", model=MODEL)

    @bentoml.api
    def predict(self, text: str) -> dict:
        return self.classifier(text)[0]
# {{/docs-fragment service}}


# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="bentoml-endpoint",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "bentoml", "torch", "transformers"
    ),
    # `bentoml serve` imports the service from this module.
    args=["bentoml", "serve", f"{Path(__file__).stem}:Classifier", "--host", "0.0.0.0", "--port", str(PORT)],
    port=PORT,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 2), scaledown_after=300),
)
# {{/docs-fragment app-env}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    deployment = flyte.serve(app_env)
    print(f"Deployed: {deployment.url}")
# {{/docs-fragment deploy}}
