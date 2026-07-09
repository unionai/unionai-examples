# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
#    "torch>=2.0",
#    "transformers>=4.38",
#    "accelerate>=1.1.0",
# ]
# main = "train_transformers"
# params = "max_epochs=10"
# ///

"""Resume Hugging Face `transformers.Trainer` training across task retries.

The `Trainer` already writes `checkpoint-<step>` directories under `output_dir`;
this example mirrors `output_dir` to the Flyte checkpoint after each epoch and
resumes with `resume_from_checkpoint` on retry.
"""

import pathlib

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_hf_trainer",
    image=flyte.Image.from_debian_base().with_pip_packages("transformers[torch]"),
)

MODEL_ID = "hf-internal-testing/tiny-random-bert"
RETRIES = 3


class ToyTextDataset(Dataset):
    """Synthetic binary classification examples."""

    def __init__(self, tokenizer, n: int = 64, max_length: int = 32):
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._texts = [f"classification example {i} with enough tokens" for i in range(n)]
        self._labels = [i % 2 for i in range(n)]

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self._tokenizer(
            self._texts[idx],
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        enc["labels"] = self._labels[idx]
        return enc


# {{docs-fragment callback}}
class FlyteTrainerCheckpointCallback(TrainerCallback):
    """Mirror the Trainer's `output_dir` to the Flyte checkpoint after each epoch."""

    def __init__(self, checkpoint: flyte.Checkpoint, output_dir: pathlib.Path) -> None:
        self._checkpoint = checkpoint
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        # Trainer callbacks are synchronous, so use save_sync
        self._checkpoint.save_sync(self._output_dir)
# {{/docs-fragment callback}}


# {{docs-fragment task}}
@env.task(retries=RETRIES)
def train_transformers(max_epochs: int = 10) -> float:
    checkpoint = flyte.ctx().checkpoint

    ckpt_dir = pathlib.Path("hf_trainer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Restore the previous attempt's checkpoint tree and find the last HF checkpoint.
    hf_resume = None
    prev = checkpoint.load_sync()
    if prev:
        hf_resume = get_last_checkpoint(str(prev))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=4,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        seed=42,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ToyTextDataset(tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[FlyteTrainerCheckpointCallback(checkpoint, ckpt_dir)],
    )
    trainer.train(resume_from_checkpoint=hf_resume)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = tokenizer(
            "classification example for inference",
            return_tensors="pt",
            truncation=True,
            max_length=32,
            padding="max_length",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        return float(logits[0, 1].item())
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_transformers, max_epochs=10)
    print(run.url)
