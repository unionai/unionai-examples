# # Fine-Tune BERT on Arabic Reviews with Multi-Node Training and Data Streaming
#
# This example demonstrates fine-tuning a BERT model on a sizable Arabic review dataset
# containing approximately 100,000 samples using Lightning and the
# [`streaming`](https://github.com/mosaicml/streaming) library for efficient, disk-optimized data loading.
# It also shows how to scale training across multiple nodes with minimal infrastructure overhead.

# {{run-on-union}}

# We preprocess the dataset into shards to enable efficient random access during training,
# and distribute the training job across multiple nodes and GPUs using the `flytekitplugins-kfpytorch` plugin,
# which uses `torchrun` under the hood.

# To start, we import the necessary libraries and set up the environment:

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import lightning
import torch
import union
from flytekit import FlyteContextManager
from flytekit.extras.accelerators import T4
from flytekitplugins.kfpytorch.task import Elastic
from flytekitplugins.wandb import wandb_init
from transformers import BertForSequenceClassification

# Since training runs across multiple nodes, we configure the setup with two nodes and six GPUs.

NUM_NODES = "2"
NUM_GPUS = "6"

# We also define the container image that includes all required dependencies for training the BERT model.

image = union.ImageSpec(
    name="arabic-bert",
    builder="union",
    packages=[
        "union==0.1.182",
        "datasets==3.3.2",
        "flytekitplugins-kfpytorch==1.15.3",
        "mosaicml-streaming==0.11.0",
        "torch==2.6.0",
        "transformers==4.49.0",
        "lightning==2.5.1",
        "cryptography<42.0.0",
        "flytekitplugins-wandb==1.15.3",
    ],
    apt_packages=["build-essential"],
)

# We then define the configuration parameters for data streaming and model training.
#
# - In the streaming config, we set the number of data loading workers,
#   the number of retry attempts for downloading shards,
#   whether to shuffle the data, and the batch size.
# - In the training config, we specify key hyperparameters such as learning rate,
#   learning rate decay (gamma), and the number of training epochs.


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


# Union.ai Artifacts allow caching of dataset and model files to speed up future runs.
# We define two artifacts: one for the dataset and one for the model.

DatasetArtifact = union.Artifact(name="arabic-reviews-shards")
ModelArtifact = union.Artifact(name="arabic-bert")

# We set the secret for authenticating with the Weights and Biases API.
# Make sure to store your API key as a secret in Union.ai.

WANDB_SECRET = union.Secret(key="wandb-api-key", env_var="WANDB_API_KEY")

# Weights and Biases entity corresponds to the user or team name in your W&B account.
# Make sure to replace it with your actual entity name.

WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"  # TODO: Replace with your W&B entity name

# We set a sensible default project name for the W&B project.
# Replace it with a project name of your choice.

WANDB_PROJECT = "bert-training"

# The function below prepares each batch of data for training by converting NumPy arrays into PyTorch tensors.
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


# To store the model and dataset artifacts, we define two tasks: `download_model` and `download_dataset`.
# The `download_model` task fetches a pretrained model from the Hugging Face Hub and caches it for use during training.
# The `download_dataset` task downloads the dataset containing 100,000 Arabic reviews,
# preprocesses it into streaming-compatible shards using `MDSWriter`, and saves it to a local directory.
# It then uploads the dataset automatically to a remote blob store via `FlyteDirectory` for efficient access during training.


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


# As part of the training pipeline, we define `BertClassifier` extending `pl.LightningModule` to wrap the
# pretrained BERT model and implement necessary training routines.


class BertClassifier(lightning.LightningModule):
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
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# To enable efficient and scalable fine-tuning of the BERT model, we set up a dedicated training task using Lightning.
# This task applies the `Elastic` strategy to distribute training across multiple nodes and GPUs and integrates the
# [Weights & Biases plugin](https://www.union.ai/docs/flyte/integrations/flytekit-plugins/wandb-plugin/) for experiment tracking.

# In the `Elastic` task configuration, we specify the number of nodes and GPUs, set the maximum number of restarts,
# and request shared memory. With this minimal setup, we can run distributed training seamlessly.

# The training data streams dynamically from a remote blob store via the `StreamingDataset` class.
# This dataset is accessed as a `FlyteDirectory`, previously prepared and uploaded in the `download_dataset` task.
# The streaming library handles shard downloads on demand, loading data into GPU memory as needed,
# which optimizes resource usage and training speed.

# > [!NOTE]
# > To learn more about how streaming works with `StreamingDataset`, check out the official
# > [streaming documentation](https://docs.mosaicml.com/projects/streaming/en/stable/getting_started/main_concepts.html).


@union.task(
    cache=True,
    container_image=image,
    task_config=Elastic(
        nnodes=int(NUM_NODES),
        nproc_per_node=int(NUM_GPUS),
        max_restarts=3,
        increase_shared_mem=True,
    ),
    requests=union.Resources(
        mem="40Gi", cpu="10", gpu=NUM_GPUS, ephemeral_storage="70Gi"
    ),
    secret_requests=[WANDB_SECRET],
    accelerator=T4,
    environment={
        "NCCL_DEBUG": "WARN",
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
    },
)
@wandb_init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    secret=WANDB_SECRET,
)
def train_bert(
    dataset_shards: union.FlyteDirectory,
    model_dir: union.FlyteDirectory,
    train_config: TrainConfig,
    streaming_config: StreamingConfig,
) -> Annotated[Optional[union.FlyteFile], ModelArtifact]:
    import os

    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import WandbLogger
    from streaming.base import StreamingDataset
    from torch.utils.data import DataLoader

    ctx = union.current_context()
    local_model_dir = os.path.join(Path(ctx.working_directory), "local_model_dir")
    FlyteContextManager.current_context().file_access.get_data(
        model_dir.remote_source, local_model_dir, is_multipart=True
    )

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
        persistent_workers=True,
    )

    wandb_logger = WandbLogger(log_model="all")

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=int(NUM_NODES),
        devices=int(NUM_GPUS),
        max_epochs=train_config.epochs,
        logger=wandb_logger,
        use_distributed_sampler=False,
    )

    trainer.fit(model, train_loader)

    if int(os.environ["RANK"]) == 0:
        model_file = os.path.join(
            union.current_context().working_directory, "bert_uncased_gpu.pt"
        )
        torch.save(model.model.state_dict(), model_file)
        return union.FlyteFile(model_file)

    return None


# > [!NOTE]
# > You can also use [Neptune Scale](https://www.union.ai/docs/flyte/integrations/flytekit-plugins/neptune-plugin/)
# > to track your experiments and model training.
# > To integrate it with PyTorch Lightning, follow the steps below:
# >
# > ```python
# > from flytekitplugins.neptune import neptune_scale_run
# >
# > @union.task(...)
# > @neptune_scale_run(
# >     project="your_project_name",
# >     secret=NEPTUNE_API_KEY,
# > )
# > def train_bert(...):
# >     ...
# >     import flytekit
# >     from lightning.pytorch.loggers import NeptuneScaleLogger
# >
# >     run = flytekit.current_context().neptune_run
# >     neptune_logger = NeptuneScaleLogger(
# >         run_id=run._run_id,
# >         api_token=run._api_token,
# >         project=run._project,
# >         resume=True,
# >   )
# >     ... # Use neptune_logger in your Trainer


# Now, let's put it all together.
# We define a workflow to download the model and dataset, and then train the BERT model on the dataset shards.


@union.workflow
def finetune_bert_on_sharded_data(
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
        streaming_config=streaming_config,
    )
