# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "flyte>=2.0.0b22",
#     "torch>=2.1.0",
#     "transformers>=4.45.0",
#     "datasets>=3.0.0",
#     "peft>=0.13.0",
#     "trl>=0.12.0",
#     "accelerate>=0.34.0",
# ]
# ///
"""
DPO fine-tuning of a small instruction-tuned LLM on Reddit TL;DR summarization
preferences, orchestrated with Flyte/Union. Runs on a single GPU (target:
16GB VRAM).

Two cacheable tasks:
  1. prepare_data — downloads + tokenizes CarperAI/openai_summarize_comparisons,
     subsamples train/eval pairs, and saves them as an HF dataset (CPU only,
     cached so re-running the pipeline doesn't redo it).
  2. train_dpo    — LoRA (peft) + trl.DPOTrainer on 1 GPU. Key hyperparameters
     are task inputs so they can be swept. Logs loss / reward margins per
     step and outputs the trained LoRA adapter.
"""

import logging
import os
import tempfile
from dataclasses import dataclass

import flyte
from flyte.io import Dir

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET_NAME = "CarperAI/openai_summarize_comparisons"
# Swap for "HuggingFaceTB/SmolLM2-1.7B-Instruct" for a larger base model.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

image = flyte.Image.from_uv_script(__file__, name="dpo-tldr-summarizer", pre=True)

# Task 1: dataset download + tokenization — CPU only, cached.
data_env = flyte.TaskEnvironment(
    name="dpo_tldr_data",
    image=image,
    resources=flyte.Resources(cpu=4, memory="8Gi", disk="20Gi"),
    cache="auto",
)

# Task 2: DPO training — 1 GPU (T4 = 16GB VRAM target).
training_env = flyte.TaskEnvironment(
    name="dpo_tldr_training",
    image=image,
    resources=flyte.Resources(cpu=8, memory="32Gi", gpu="T4:1", disk="50Gi"),
    cache="auto",
)

# Driver — wires the two tasks together.
pipeline_env = flyte.TaskEnvironment(
    name="dpo_tldr_pipeline",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    depends_on=[data_env, training_env],
)


@dataclass
class DPOHyperparameters:
    model_name: str = DEFAULT_MODEL_NAME
    learning_rate: float = 5e-5
    beta: float = 0.1
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_prompt_length: int = 512
    max_length: int = 1024


# ---------------------------------------------------------------------------
# 1. prepare_data
# ---------------------------------------------------------------------------
@data_env.task(cache="auto")
async def prepare_data(
    model_name: str = DEFAULT_MODEL_NAME,
    max_train_samples: int = 8000,
    max_eval_samples: int = 500,
    max_prompt_length: int = 512,
    max_length: int = 1024,
    seed: int = 42,
) -> Dir:
    """
    Download CarperAI/openai_summarize_comparisons (prompt/chosen/rejected),
    tokenize with the target model's tokenizer to drop pairs that would be
    truncated during DPO training, subsample, split into train/eval, and
    save as an HF dataset (flyte.io.Dir).
    """
    from datasets import DatasetDict, get_dataset_split_names, load_dataset
    from transformers import AutoTokenizer

    log.info(f"Loading dataset: {DATASET_NAME}")
    splits = get_dataset_split_names(DATASET_NAME)
    train_split = "train" if "train" in splits else splits[0]
    eval_split = next(
        (s for s in splits if s != train_split and ("valid" in s or "test" in s)),
        None,
    )

    train_ds = load_dataset(DATASET_NAME, split=train_split)
    if eval_split is not None:
        eval_ds = load_dataset(DATASET_NAME, split=eval_split)
    else:
        split = train_ds.train_test_split(test_size=max_eval_samples, seed=seed)
        train_ds, eval_ds = split["train"], split["test"]

    required_cols = {"prompt", "chosen", "rejected"}
    missing = required_cols - set(train_ds.column_names)
    if missing:
        raise ValueError(
            f"{DATASET_NAME} is missing expected columns {missing}; found {train_ds.column_names}"
        )
    train_ds = train_ds.select_columns(list(required_cols))
    eval_ds = eval_ds.select_columns(list(required_cols))

    train_ds = train_ds.shuffle(seed=seed).select(range(min(max_train_samples, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=seed).select(range(min(max_eval_samples, len(eval_ds))))

    log.info(f"Tokenizing with {model_name} to filter over-length pairs")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fits(example):
        prompt_len = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
        chosen_len = len(tokenizer(example["chosen"], add_special_tokens=False)["input_ids"])
        rejected_len = len(tokenizer(example["rejected"], add_special_tokens=False)["input_ids"])
        return (
            prompt_len <= max_prompt_length
            and prompt_len + chosen_len <= max_length
            and prompt_len + rejected_len <= max_length
        )

    train_ds = train_ds.filter(fits)
    eval_ds = eval_ds.filter(fits)

    dataset = DatasetDict({"train": train_ds, "eval": eval_ds})
    output_dir = os.path.join(tempfile.mkdtemp(), "dpo_dataset")
    dataset.save_to_disk(output_dir)
    log.info(f"Dataset ready: {len(dataset['train'])} train pairs, {len(dataset['eval'])} eval pairs")

    return await Dir.from_local(output_dir)


# ---------------------------------------------------------------------------
# 2. train_dpo
# ---------------------------------------------------------------------------
@training_env.task
async def train_dpo(
    data_dir: Dir,
    model_name: str = DEFAULT_MODEL_NAME,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    num_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_prompt_length: int = 512,
    max_length: int = 1024,
) -> Dir:
    """
    Fine-tune `model_name` with DPO + LoRA on the prepared preference dataset.
    trl.DPOTrainer manages the frozen reference model internally (base model
    with the LoRA adapter disabled) when a `peft_config` is supplied.
    Returns the trained LoRA adapter as a flyte.io.Dir.
    """
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    log.info(f"DPO training: model={model_name} lora_r={lora_r} beta={beta} lr={learning_rate}")

    local_data = await data_dir.download()
    dataset = load_from_disk(local_data)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Attention projection layers only, per the target rank/module spec.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    class MetricsLoggerCallback(TrainerCallback):
        """Logs DPO loss / reward margins / implicit accuracy per step via Flyte's task logs."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            log.info(
                "step=%d epoch=%.2f loss=%.4f rewards/margins=%.4f rewards/accuracies=%.4f",
                state.global_step,
                logs.get("epoch", 0.0),
                logs["loss"],
                logs.get("rewards/margins", float("nan")),
                logs.get("rewards/accuracies", float("nan")),
            )

    output_dir = os.path.join(tempfile.mkdtemp(), "dpo_checkpoints")
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[MetricsLoggerCallback()],
    )

    log.info("Starting DPO training...")
    trainer.train()
    log.info("DPO training complete.")

    adapter_dir = os.path.join(tempfile.mkdtemp(), "dpo_adapter")
    trainer.save_model(adapter_dir)  # model is a PeftModel -> saves adapter weights only
    tokenizer.save_pretrained(adapter_dir)

    return await Dir.from_local(adapter_dir)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
@pipeline_env.task
async def dpo_tldr_pipeline(hp: DPOHyperparameters = DPOHyperparameters()) -> Dir:
    """Prepare the TL;DR preference dataset, then run DPO training and return the LoRA adapter."""
    data_dir = await prepare_data(
        model_name=hp.model_name,
        max_prompt_length=hp.max_prompt_length,
        max_length=hp.max_length,
    )
    return await train_dpo(
        data_dir=data_dir,
        model_name=hp.model_name,
        learning_rate=hp.learning_rate,
        beta=hp.beta,
        num_epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        gradient_accumulation_steps=hp.gradient_accumulation_steps,
        lora_r=hp.lora_r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        max_prompt_length=hp.max_prompt_length,
        max_length=hp.max_length,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(dpo_tldr_pipeline, hp=DPOHyperparameters())
    print(f"Run URL: {run.url}")
