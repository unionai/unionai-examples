# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "litserve",
#    "torch",
#    "transformers",
# ]
# ///

"""Serve a Hugging Face model with LitServe on a generic AppEnvironment."""

import sys
from pathlib import Path

import flyte
import flyte.app
import litserve as ls
from transformers import pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
PORT = 8080

file_name = Path(__file__).name

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="litserve-endpoint",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "litserve", "torch", "transformers"
    ),
    args=["python", file_name, "--server"],
    port=PORT,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="10Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 2), scaledown_after=300),
)
# {{/docs-fragment app-env}}


# {{docs-fragment server}}
# LitServe pickles the API class into its worker processes, so define it at module level.
class Classifier(ls.LitAPI):
    def setup(self, device):
        self.classifier = pipeline("text-classification", model=MODEL, device=device)

    def decode_request(self, request):
        return request["text"]

    def predict(self, text):
        return self.classifier(text)[0]


def serve() -> None:
    server = ls.LitServer(Classifier(), accelerator="auto")
    server.run(host="0.0.0.0", port=PORT)
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
