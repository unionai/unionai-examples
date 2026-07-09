# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
#    "torch>=2.0",
#    "lightning>=2.0",
# ]
# main = "train_lightning"
# params = "max_epochs=10"
# ///

"""Resume PyTorch Lightning training across task retries with `flyte.Checkpoint`.

Lightning already writes `last.ckpt` to a local directory via `ModelCheckpoint`;
this example mirrors that directory to the Flyte checkpoint after each epoch and
resumes from the newest `last.ckpt` on retry.
"""

import pathlib

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_lightning",
    image=flyte.Image.from_debian_base().with_pip_packages("lightning"),
)

FEATURES = 16
RETRIES = 3


# {{docs-fragment callback}}
class FlyteLightningCheckpointCallback(ModelCheckpoint):
    """A `ModelCheckpoint` that mirrors `dirpath` to the Flyte checkpoint after each epoch."""

    def __init__(self, flyte_checkpoint: flyte.Checkpoint, *, dirpath: str | pathlib.Path, **kwargs) -> None:
        super().__init__(dirpath=str(dirpath), **kwargs)
        self._flyte_checkpoint = flyte_checkpoint

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self.dirpath:
            # Lightning callbacks are synchronous, so use save_sync
            self._flyte_checkpoint.save_sync(pathlib.Path(self.dirpath))
# {{/docs-fragment callback}}


class TinyModule(L.LightningModule):
    def __init__(self, lr: float = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Linear(FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.lr)


def make_loader(batch: int = 32, batches: int = 8) -> DataLoader:
    g = torch.Generator().manual_seed(42)
    x = torch.randn(batches * batch, FEATURES, generator=g)
    y = torch.randn(batches * batch, 1, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True)


# {{docs-fragment task}}
@env.task(retries=RETRIES)
def train_lightning(max_epochs: int = 10) -> float:
    checkpoint = flyte.ctx().checkpoint

    ckpt_dir = pathlib.Path("pl_checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Restore the previous attempt's checkpoint tree and find the newest last.ckpt.
    resume_ckpt = None
    prev = checkpoint.load_sync()
    if prev:
        last = flyte.latest_checkpoint(prev)
        if last:
            resume_ckpt = str(last)

    model = TinyModule()
    mc = FlyteLightningCheckpointCallback(
        checkpoint,
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
        save_top_k=1,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        callbacks=[mc],
        enable_progress_bar=True,
        logger=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, make_loader(), ckpt_path=resume_ckpt)

    with torch.no_grad():
        return float(model(torch.ones(1, FEATURES)).squeeze().item())
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_lightning, max_epochs=10)
    print(run.url)
