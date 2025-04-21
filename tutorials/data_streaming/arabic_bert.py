# # Streaming Data for BERT Training
#
# This example demonstrates how to train a BERT model on a large Arabic text dataset using
# PyTorch Lightning and the [`streaming`](https://github.com/mosaicml/streaming) library from MosaicML.

# {{run-on-union}}

# The dataset is preprocessed into shards to enable efficient random access during training.
# The training job is distributed across multiple GPUs using the `flytekitplugins-kfpytorch` plugin,
# which leverages `torchrun` under the hood for multi-process training.

# To get started, import the necessary libraries and set up the environment:

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import pytorch_lightning as pl
import torch
import union
from flytekit.extras.accelerators import T4
from flytekitplugins.kfpytorch.task import Elastic
from transformers import BertForSequenceClassification

# Set the number of nodes and GPUs to be used for training.

NUM_NODES = "1"
NUM_GPUS = "2"

# Define the container image to be used for the tasks.
# This image includes all the necessary dependencies for training the BERT model.

image = union.ImageSpec(
    name="arabic-bert",
    builder="union",
    packages=[
        "union==0.1.173",
        "datasets==3.3.2",
        "flytekitplugins-kfpytorch==1.15.3",
        "mosaicml-streaming==0.11.0",
        "torch==2.6.0",
        "transformers==4.49.0",
        "wandb==0.19.8",
        "pytorch-lightning==2.5.1",
    ],
)

# Define configuration parameters for both data streaming and model training.
#
# - The streaming configuration specifies the number of data loading workers, the number of retry attempts for downloading shards,
#   whether to shuffle the data during training, and the batch size.
# - The training configuration defines key training hyperparameters such as learning rate,
#   learning rate decay (gamma), and number of training epochs.


@dataclass
class StreamingConfig:
    num_workers: int = 2
    download_retry: int = 2
    shuffle: bool = True
    batch_size: int = 8


@dataclass
class TrainConfig:
    lr: float = 1.0
    gamma: float = 0.7
    epochs: int = 2


# Define the artifacts for the dataset and model.
# These artifacts enable caching of the dataset and model files for future runs.

DatasetArtifact = union.Artifact(name="arabic-reviews-shards")
ModelArtifact = union.Artifact(name="arabic-bert")

# Set the secret for authenticating with the Weights and Biases API.
# Make sure to request or store your API key as a secret in Union.

WANDB_API_KEY = "wandb-api-key"


# Define the custom collate function for the `DataLoader`.
# This function prepares each batch of data for training by converting NumPy arrays into PyTorch tensors.
# It also ensures that data is correctly formatted and writable before conversion, which is especially
# important when working with memory-mapped arrays or data streaming.


def collate_fn(batch):
    import torch

    collated_batch = {}
    for key in batch[0].keys():
        if key == "labels":
            collated_batch[key] = torch.tensor([item[key] for item in batch])
        else:
            # Ensure arrays are writable before conversion
            tensors = []
            for item in batch:
                value = item[key]
                if hasattr(value, "flags") and not value.flags.writeable:
                    value = value.copy()
                tensors.append(torch.tensor(value))
            collated_batch[key] = torch.stack(tensors)
    return collated_batch


# Define the tasks for downloading the model and dataset.
# The `download_model` task fetches a pretrained model from the Hugging Face Hub and caches it for use during training.
# The `download_dataset` task downloads the dataset containing 100,000 Arabic reviews,
# preprocesses it into streaming-compatible shards using `MDSWriter`, and saves it to a local directory.
# The dataset is then automatically uploaded to a remote blob store using `FlyteDirectory` for efficient access during training.


@union.task(cache=True, requests=union.Resources(mem="5Gi"), container_image=image)
def download_model(model_name: str) -> Annotated[union.FlyteDirectory, ModelArtifact]:
    from huggingface_hub import snapshot_download

    ctx = union.current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    snapshot_download(model_name, local_dir=cached_model_dir)
    return cached_model_dir


@union.task(
    cache=True, container_image=image, requests=union.Resources(cpu="3", mem="3Gi")
)
def download_dataset(
    dataset: str, model_dir: union.FlyteDirectory
) -> Annotated[union.FlyteDirectory, DatasetArtifact]:
    from datasets import ClassLabel, load_dataset
    from streaming.base import MDSWriter
    from transformers import AutoTokenizer

    loaded_dataset = load_dataset(dataset, split="train")
    loaded_dataset = loaded_dataset.shuffle(seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_dir.download())

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.cast_column(
        "label", ClassLabel(names=["Positive", "Negative", "Mixed"])
    )
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    tokenized_dataset.set_format("numpy")

    local_dir = os.path.join(union.current_context().working_directory, "mds_shards")
    os.makedirs(local_dir, exist_ok=True)

    # Use MDSWriter to write the dataset to local directory
    with MDSWriter(
        out=local_dir,
        columns={
            "input_ids": "ndarray",
            "attention_mask": "ndarray",
            "token_type_ids": "ndarray",
            "labels": "int64",
        },
        size_limit="100kb",
    ) as out:
        for i in range(len(tokenized_dataset)):
            out.write(
                {k: tokenized_dataset[i][k] for k in tokenized_dataset.column_names}
            )

    return union.FlyteDirectory(local_dir)


# Define the BERT classifier model using PyTorch Lightning.
# This module wraps Hugging Face’s `BertForSequenceClassification` model in a PyTorch Lightning module.
# It supports multi-class classification and is configured with an adaptive learning rate scheduler for training stability.


class BertClassifier(pl.LightningModule):
    def __init__(self, model_dir: str, learning_rate: float, gamma: float):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model_dir, num_labels=3
        )
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.save_hyperparameters()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# Set up a training task to fine-tune the BERT model using PyTorch Lightning.
# This task leverages the `Elastic` strategy to distribute training across 2 GPUs on a single node,
# and uses `WandbLogger` to log metrics to Weights & Biases for experiment tracking.

# The training data is streamed from a remote blob store using the `StreamingDataset` class.
# The dataset is provided as a `FlyteDirectory`, which was created and uploaded in the earlier `download_dataset` task.
# The `streaming` library downloads shards on demand and loads them into GPU memory as needed, enabling efficient training at scale.


@union.task(
    cache=True,
    container_image=image,
    task_config=Elastic(
        nnodes=int(NUM_NODES),
        nproc_per_node=int(NUM_GPUS),
        max_restarts=3,
        start_method="fork",
    ),
    requests=union.Resources(
        mem="40Gi", cpu="10", gpu=NUM_GPUS, ephemeral_storage="15Gi"
    ),
    secret_requests=[union.Secret(key=WANDB_API_KEY, env_var="WANDB_API_KEY")],
    accelerator=T4,
    environment={
        "NCCL_DEBUG": "WARN",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
    },
    shared_memory=True,
)
def train_bert(
    dataset_shards: union.FlyteDirectory,
    model_dir: union.FlyteDirectory,
    train_config: TrainConfig,
    wandb_entity: str,
    streaming_config: StreamingConfig,
) -> Annotated[Optional[union.FlyteFile], ModelArtifact]:
    import os

    import pytorch_lightning as pl
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from streaming.base import StreamingDataset
    from torch.utils.data import DataLoader

    local_model_dir = model_dir.download()
    model = BertClassifier(local_model_dir, train_config.lr, train_config.gamma)

    dataset = StreamingDataset(
        remote=dataset_shards.remote_source,
        batch_size=streaming_config.batch_size,
        download_retry=streaming_config.download_retry,
        shuffle=streaming_config.shuffle,
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=streaming_config.batch_size,
        collate_fn=collate_fn,
        num_workers=streaming_config.num_workers,
    )

    wandb_logger = WandbLogger(
        entity=wandb_entity,
        project="bert-training",
        name=f"bert-training-rank-{os.environ['RANK']}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices="auto",
        max_epochs=train_config.epochs,
        logger=wandb_logger,
        use_distributed_sampler=False,
    )

    trainer.fit(model, train_loader)

    # Save model only from rank 0
    if int(os.environ["RANK"]) == 0:
        model_file = os.path.join(
            union.current_context().working_directory, "bert_uncased_gpu.pt"
        )
        torch.save(model.model.state_dict(), model_file)
        wandb.finish()
        return union.FlyteFile(model_file)

    return None


# Define the workflow for downloading the model, dataset, and training the BERT model.
# The workflow orchestrates the execution of the tasks and ensures that the model and dataset are available for training.


@union.workflow
def finetune_bert_on_sharded_data(
    wandb_entity: str,
    dataset_name: str = "arbml/arabic_100k_reviews",
    model_name: str = "bert-base-uncased",
    train_config: TrainConfig = TrainConfig(),
    streaming_config: StreamingConfig = StreamingConfig(),
) -> Optional[union.FlyteFile]:
    model = download_model(model_name=model_name)
    dataset_shards = download_dataset(dataset=dataset_name, model_dir=model)
    return train_bert(
        dataset_shards=dataset_shards,
        model_dir=model,
        train_config=train_config,
        wandb_entity=wandb_entity,
        streaming_config=streaming_config,
    )
