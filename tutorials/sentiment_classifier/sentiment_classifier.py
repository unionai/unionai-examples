"""Runs ML workflow with wandb integration.

1. Download dataset
2. Download model weights
3. Train model and upload data to wandb
4. Configure artifact metadata in wandb
5. Run a second workflow with another model and compare models

Create a secret for wandb :
```
unionai create secret wandb_api_key
```

Usage:
```
unionai run --remote tutorials/sentiment_classifier/sentiment_classifier.py main
```
"""

from pathlib import Path
import tarfile
import os
from flytekit import task, workflow, current_context, ImageSpec, Secret, Resources
from flytekit.extras import accelerators
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile


image_spec = ImageSpec(
    packages=[
        "accelerate==0.30.1",
        "datasets==2.19.2",
        "numpy==1.26.4",
        "transformers==4.41.2",
        "wandb==0.17.0",
        "torch==2.0.1",
    ],
    cuda="11.8",
)


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


@task(
    container_image=image_spec,
    secret_requests=[Secret(key="wandb_api_key")],
    requests=Resources(cpu="2", mem="12Gi", gpu="1"),
    accelerator=accelerators.T4,
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

    if torch.cuda.is_available():
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
        evaluation_strategy="epoch",
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


@workflow
def main(
    model: str = "distilbert-base-uncased",
    wandb_project: str = "unionai-serverless-demo",
    n_epochs: int = 30,
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
 