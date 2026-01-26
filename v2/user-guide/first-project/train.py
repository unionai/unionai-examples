# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte==2.0.0b44",
#    "torch>=2.0.0",
#    "transformers>=4.35.0",
#    "datasets>=2.14.0",
#    "accelerate>=0.24.0",
# ]
# main = "training_pipeline"
# params = "max_samples=1000, epochs=1"
# ///

"""
Model Training Pipeline

Fine-tunes DistilGPT-2 on the wikitext-2 dataset for text generation.
The trained model is saved as an artifact that can be used by the serving app.
"""

# {{docs-fragment imports}}
import os
import tempfile

import flyte
from flyte.io import Dir, File

# {{/docs-fragment imports}}

# {{docs-fragment training-env}}
training_env = flyte.TaskEnvironment(
    name="model-training",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    cache="auto",
)
# {{/docs-fragment training-env}}


# {{docs-fragment prepare-data}}
@training_env.task
async def prepare_data(max_samples: int = 1000) -> Dir:
    """
    Load and tokenize the wikitext-2 dataset.

    Args:
        max_samples: Maximum number of training samples to use.

    Returns:
        Directory containing tokenized train and validation datasets.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )

    # Filter out empty strings and tokenize
    train_data = dataset["train"].filter(lambda x: len(x["text"].strip()) > 10)
    val_data = dataset["validation"].filter(lambda x: len(x["text"].strip()) > 10)

    # Limit samples for faster training
    if max_samples > 0:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 10, len(val_data))))

    print(f"Tokenizing {len(train_data)} training samples...")
    train_tokenized = train_data.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    print(f"Tokenizing {len(val_data)} validation samples...")
    val_tokenized = val_data.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Save datasets to disk for Flyte serialization
    data_dir = tempfile.mkdtemp()
    train_tokenized.save_to_disk(os.path.join(data_dir, "train"))
    val_tokenized.save_to_disk(os.path.join(data_dir, "validation"))

    # Save tokenizer name as metadata
    with open(os.path.join(data_dir, "tokenizer_name.txt"), "w") as f:
        f.write("distilgpt2")

    print(f"Saved datasets to {data_dir}")
    return await Dir.from_local(data_dir)


# {{/docs-fragment prepare-data}}


# {{docs-fragment fine-tune}}
@training_env.task
async def fine_tune_model(data_dir: Dir, epochs: int = 1) -> File:
    """
    Fine-tune DistilGPT-2 on the prepared dataset.

    Args:
        data_dir: Directory containing tokenized datasets from prepare_data.
        epochs: Number of training epochs.

    Returns:
        File object pointing to the saved model archive.
    """
    import torch
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    # Download data directory from remote storage
    local_data_path = await data_dir.download()

    # Load datasets from disk
    train_data = load_from_disk(os.path.join(local_data_path, "train"))
    val_data = load_from_disk(os.path.join(local_data_path, "validation"))

    # Load tokenizer name from metadata
    with open(os.path.join(local_data_path, "tokenizer_name.txt")) as f:
        tokenizer_name = f.read().strip()

    # Detect device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create temporary directory for training outputs
    with tempfile.TemporaryDirectory() as output_dir:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),  # Use FP16 on GPU
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        # Save the final model
        model_save_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_save_dir, "model")
        print(f"Saving model to {model_path}...")
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)

        # Create a tar archive of the model
        import shutil

        archive_path = os.path.join(model_save_dir, "model.tar.gz")
        shutil.make_archive(
            os.path.join(model_save_dir, "model"), "gztar", model_save_dir, "model"
        )

        print(f"Model archived to {archive_path}")
        return await File.from_local(archive_path)


# {{/docs-fragment fine-tune}}


# {{docs-fragment training-pipeline}}
@training_env.task
async def training_pipeline(max_samples: int = 1000, epochs: int = 1) -> File:
    """
    Main training pipeline that orchestrates data preparation and model fine-tuning.

    Args:
        max_samples: Maximum number of training samples to use.
        epochs: Number of training epochs.

    Returns:
        File object pointing to the trained model archive.
    """
    print(f"Starting training pipeline: max_samples={max_samples}, epochs={epochs}")

    # Prepare data
    data_dir = await prepare_data(max_samples)

    # Fine-tune model
    model_file = await fine_tune_model(data_dir, epochs)

    print("Training pipeline complete!")
    return model_file


# {{/docs-fragment training-pipeline}}

# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(training_pipeline, max_samples=1000, epochs=1)
    print(f"Training run URL: {run.url}")
    run.wait()
    print(f"Training complete! Model file: {run.outputs()}")
# {{/docs-fragment main}}
