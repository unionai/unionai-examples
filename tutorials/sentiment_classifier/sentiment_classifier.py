# # Sentiment Classification with Language Models
#
# This tutorial demonstrates how to fine-tune a pre-trained language model to
# classify the sentiment of IMDB movie reviews. We're going to use the
# `transformers` library and the `imdb` dataset to classify the movie review
# sentiment.

# {{run-on-union}}

# ## Overview
#
# The power of language models lies in their flexibility – as long as you
# operate in the same token space as a pre-trained model, you can leverage the
# patterns learned from a much wider data distribution than you could learn from
# just a small data domain.
#
# In this example, we're going to fine-tune the [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) model on the [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) to classify the sentiment of movie reviews.
#
# We'll start by importing the workflow dependencies:

from pathlib import Path
import tarfile
import os
from flytekit import task, workflow, current_context, ImageSpec, Secret, Resources
from flytekit.extras import accelerators
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

# ## Defining the container image
#
# We'll define the container image that will be used to run the workflow with
# the `ImageSpec` object:

image_spec = ImageSpec(
    name="sentiment_classifier",
    packages=[
        "accelerate==0.33.0",
        "datasets==2.20.0",
        "numpy==1.26.4",
        "transformers==4.44.0",
        "wandb==0.17.6",
        "torch==2.4.0",
    ],
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)

# We've pinned the versions of the package dependencies to ensure reproducibility.
# Under the hood, Union will build the container image so we don't have to
# worry about writing a `Dockerfile`.
#
# ## Downloading the dataset and model
#
# Next, we download the dataset. Specifying `cache=True` in the `@task`
# definition makes sure that we don't waste compute resources downloading the
# data multiple times:


@task(
    container_image=image_spec,
    cache=True,
    cache_version="v8",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_dataset() -> FlyteDirectory:
    """Download and pre-cache the IMDB dataset."""
    from datasets import load_dataset

    working_dir = Path(current_context().working_directory)
    dataset_cache_dir = working_dir / "dataset_cache"
    load_dataset("imdb", cache_dir=dataset_cache_dir)

    return dataset_cache_dir


# Then we'll do the same for the model:


@task(
    container_image=image_spec,
    cache=True,
    cache_version="v8",
    requests=Resources(cpu="2", mem="2Gi"),
)
def download_model(model: str) -> FlyteDirectory:
    """Download and pre-cache the model weights."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    working_dir = Path(current_context().working_directory)
    model_cache_dir = working_dir / "model_cache"

    AutoTokenizer.from_pretrained(model, cache_dir=model_cache_dir)
    AutoModelForSequenceClassification.from_pretrained(model, cache_dir=model_cache_dir)
    return model_cache_dir


# ## Fine-tuning the model
#
# Now we're ready to fine-tune the model using the dataset and model from the previous
# steps. The task below does the following:
#
# 1. Loads the dataset and model.
# 2. Tokenizes the dataset.
# 3. Initializes a weights and biases session to track the training process.
# 4. Trains the model based on the number of epochs (`n_epochs`) specified.
# 5. Compresses the model to a tarfile and saves it to the specified path.


@task(
    container_image=image_spec,
    secret_requests=[Secret(key="wandb_api_key")],
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
    accelerator=accelerators.A100,
)
def train_model(
    model_name: str,
    n_epochs: int,
    wandb_project: str,
    model_cache_dir: FlyteDirectory,
    dataset_cache_dir: FlyteDirectory,
) -> tuple[str, FlyteFile]:
    """Train a sentiment classifier using the imdb dataset."""
    from datasets import load_dataset
    import numpy as np
    import torch

    import wandb
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        pipeline,
    )

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "models"

    model_cache_dir.download()
    dataset_cache_dir.download()

    # load the dataset and model
    dataset = load_dataset("imdb", cache_dir=dataset_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
        cache_dir=model_cache_dir,
    )
    model = model.to("cuda")

    def tokenizer_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Use a small subset such that finetuning completes
    small_train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(500)).map(tokenizer_function)
    )
    small_eval_dataset = (
        dataset["test"].shuffle(seed=42).select(range(100)).map(tokenizer_function)
    )

    # define evaluation metric
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": np.mean(predictions == labels)}

    # set wandb environment variables
    os.environ["WANDB_API_KEY"] = ctx.secrets.get(key="wandb_api_key")
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_LOG_MODEL"] = "end"

    run = wandb.init(project=wandb_project, save_code=True, tags=[model_name])

    training_args = TrainingArguments(
        output_dir=train_dir,
        eval_strategy="epoch",
        num_train_epochs=n_epochs,
        report_to="wandb",
        logging_steps=50,
    )

    # start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    wandb.finish()

    # save the inference pipeline
    wandb_url = run.get_url()
    inference_path = working_dir / "inference_pipe"
    inference_pipe = pipeline("text-classification", tokenizer=tokenizer, model=model)
    inference_pipe.save_pretrained(inference_path)

    # compress the inference pipeline
    inference_path_compressed = working_dir / "inference_pipe.tar.gz"
    with tarfile.open(inference_path_compressed, "w:gz") as tar:
        tar.add(inference_path, arcname="")

    return wandb_url, inference_path_compressed


# ## Creating the workflow
#
# We can put all of these tasks together into a workflow:


@workflow
def main(
    model: str = "distilbert-base-uncased",
    wandb_project: str = "unionai-serverless-demo",
    n_epochs: int = 5,
) -> tuple[str, FlyteFile]:
    """IMDB sentiment classifier workflow."""
    dataset_cache_dir = download_dataset()
    model_cache_dir = download_model(model=model)
    return train_model(
        model_name=model,
        n_epochs=n_epochs,
        wandb_project=wandb_project,
        model_cache_dir=model_cache_dir,
        dataset_cache_dir=dataset_cache_dir,
    )


# Each task is actually running in its own container, but Union takes care of
# storing the intermediate outputs and passing them between tasks.
#
# ## Trying out different models
#
# Now that you've run the fine-tuning workflow once, you can try out different
# models by passing in a different model name to the `model` argument, which can
# be supplied to the `--model` flag when you invoke `union run`. For example,
# you can try out the `google-bert/bert-base-uncased` model, or any text
# [classification model](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending) available on HuggingFace hub.
