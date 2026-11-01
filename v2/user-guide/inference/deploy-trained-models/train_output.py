# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "torch",
#    "transformers",
#    "accelerate",
#    "joblib",
#    "scikit-learn",
# ]
# ///

"""Train models in tasks and return them as task outputs."""

import os

import flyte
import joblib
from flyte.io import Dir
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# {{docs-fragment finetune}}
env = flyte.TaskEnvironment(
    name="training",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch", "transformers", "accelerate", "joblib", "scikit-learn"
    ),
    resources=flyte.Resources(cpu="4", memory="32Gi", gpu="L40s:1", disk="100Gi"),
)


@env.task
async def finetune(base_model: str) -> Dir:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # Add your training loop here.

    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    return await Dir.from_local("model")
# {{/docs-fragment finetune}}


# {{docs-fragment train-classifier}}
@env.task
async def train_classifier() -> Dir:
    features, labels = make_classification(n_samples=500, n_features=4, random_state=0)
    model = LogisticRegression().fit(features, labels)

    os.makedirs("classifier", exist_ok=True)
    joblib.dump(model, "classifier/model.joblib")
    return await Dir.from_local("classifier")
# {{/docs-fragment train-classifier}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(finetune, base_model="Qwen/Qwen3-0.6B")
    print(run.url)
