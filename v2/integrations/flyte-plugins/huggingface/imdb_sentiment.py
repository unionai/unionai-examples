# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-huggingface",
#    "datasets",
#    "transformers",
#    "torch",
#    "scikit-learn",
#    "accelerate",
#    "numpy",
# ]
# main = "main"
# params = ""
# ///
"""Fine-tune DistilBERT on IMDB, sourcing the dataset straight from the Hub.

Every dataset that crosses a task boundary here is a `datasets.Dataset`. The
plugin handles the Parquet on both sides, so no task writes a file by hand.
"""

# {{docs-fragment env}}
import flyte

image = flyte.Image.from_uv_script(__file__, name="imdb-sentiment", pre=True)

cpu_env = flyte.TaskEnvironment(
    name="imdb_sentiment_cpu",
    image=image,
    resources=flyte.Resources(cpu="4", memory="8Gi"),
)

gpu_env = flyte.TaskEnvironment(
    name="imdb_sentiment_gpu",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu=1),
    # Only needed for gated or private repos; IMDB is public.
    secrets=[flyte.Secret(key="huggingface-token", as_env_var="HF_TOKEN")],
)

REPO = "stanfordnlp/imdb"
CONFIG = "plain_text"
MODEL = "distilbert-base-uncased"

# Shared across every run in this project. The first run downloads IMDB from the
# Hub; every run after that reads these Parquet shards instead.
CACHE_ROOT = "s3://my-bucket/flyte-hf-cache"
# {{/docs-fragment env}}


# {{docs-fragment prepare}}
import datasets


@cpu_env.task(cache="auto")
async def subsample(ds: datasets.Dataset, n_rows: int, seed: int = 42) -> datasets.Dataset:
    """Take a reproducible random subset, so the pipeline is cheap to iterate on.

    flatten_indices() materializes the shuffled selection into a real table.
    Skip it and shuffle/select leave only an index mapping, which the encoder
    does not serialize -- the task would hand the full 25,000 rows downstream.
    """
    return ds.shuffle(seed=seed).select(range(min(n_rows, len(ds)))).flatten_indices()
# {{/docs-fragment prepare}}


# {{docs-fragment tokenize}}
@cpu_env.task(cache="auto")
async def tokenize(ds: datasets.Dataset, max_length: int = 256) -> datasets.Dataset:
    """Tokenize on CPU, once, and hand the result on as a dataset.

    The returned dataset carries list-valued `input_ids` and `attention_mask`
    columns. Those survive the Parquet round trip, so the GPU task receives them
    ready to train on and never loads a tokenizer.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    return ds.map(encode, batched=True, remove_columns=["text"])
# {{/docs-fragment tokenize}}


# {{docs-fragment train}}
import flyte.io


@gpu_env.task
async def finetune(
    train_ds: datasets.Dataset,
    eval_ds: datasets.Dataset,
    epochs: float = 1.0,
    lr: float = 5e-5,
    batch_size: int = 16,
) -> flyte.io.Dir:
    import tempfile

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    out_dir = tempfile.mkdtemp()

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(out_dir)

    return await flyte.io.Dir.from_local(out_dir)
# {{/docs-fragment train}}


# {{docs-fragment evaluate}}
@gpu_env.task(report=True)
async def score_stream(
    model_dir: flyte.io.Dir,
    ds: datasets.IterableDataset,
    batch_size: int = 32,
) -> float:
    """Score held-out reviews by streaming them, never holding the split in memory.

    The caller passes the same value it gave `tokenize`, a `datasets.Dataset`.
    It arrives here as an IterableDataset purely because that is the annotation,
    so rows are pulled from Parquet a batch at a time and peak memory is one
    batch whether the split holds 1,000 rows or 25,000.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local = await model_dir.download()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(local).eval()

    correct = seen = 0
    batch: list[dict] = []

    def flush(rows):
        nonlocal correct, seen
        if not rows:
            return
        enc = tokenizer(
            [r["text"] for r in rows], truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            preds = model(**enc).logits.argmax(dim=-1).tolist()
        correct += sum(int(p == r["label"]) for p, r in zip(preds, rows))
        seen += len(rows)

    for row in ds:
        batch.append(row)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    flush(batch)

    accuracy = correct / seen
    await flyte.report.replace.aio(
        f"<h2>Held-out accuracy</h2><p>{accuracy:.1%} over {seen} streamed reviews.</p>"
    )
    await flyte.report.flush.aio()
    return accuracy
# {{/docs-fragment evaluate}}


# {{docs-fragment main}}
import asyncio

from flyteplugins.huggingface.datasets import from_hf


@cpu_env.task
async def main(train_rows: int = 4_000, eval_rows: int = 1_000) -> float:
    # Both references point at the same cache_root, so the two splits download
    # once and every later run of this pipeline skips the Hub entirely.
    train_src = from_hf(REPO, name=CONFIG, split="train", cache_root=CACHE_ROOT)
    test_src = from_hf(REPO, name=CONFIG, split="test", cache_root=CACHE_ROOT)

    train_raw, eval_raw = await asyncio.gather(
        subsample(train_src, train_rows),
        subsample(test_src, eval_rows),
    )
    train_tok, eval_tok = await asyncio.gather(tokenize(train_raw), tokenize(eval_raw))

    model_dir = await finetune(train_tok, eval_tok)

    # eval_raw is a datasets.Dataset. score_stream annotates the same value as an
    # IterableDataset and therefore receives it as a stream -- same bytes in
    # storage, different view. Note it scores the *shuffled* sample rather than
    # the head of the raw split: IMDB's test split is ordered by label, so the
    # first N rows are all negative and would score meaninglessly high.
    return await score_stream(model_dir, eval_raw)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment main}}
