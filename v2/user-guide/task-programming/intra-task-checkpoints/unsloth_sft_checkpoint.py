# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.5.0",
#    "torch>=2.0",
#    "unsloth",
#    "trl",
#    "datasets",
#    "transformers",
#    "accelerate",
# ]
# main = "train_unsloth_sft"
# params = "max_epochs=10"
# ///

"""Resume Unsloth LoRA fine-tuning (`trl.SFTTrainer`) across task retries.

Same pattern as the Hugging Face `Trainer` example: mirror `output_dir` to the
Flyte checkpoint after each epoch and resume with `resume_from_checkpoint`.
Unsloth requires an NVIDIA, AMD, or Intel GPU, so the task requests one.
"""

import pathlib

try:
    import unsloth  # noqa: F401
except NotImplementedError:
    # Unsloth raises on unsupported hardware at import time (e.g. Apple Silicon);
    # the task itself runs on a GPU worker.
    pass

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

import flyte

env = flyte.TaskEnvironment(
    name="checkpoint_unsloth_sft",
    image=flyte.Image.from_debian_base().with_pip_packages("unsloth"),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu="L4:1"),
)

MODEL_NAME = "unsloth/Llama-3.2-1B-bnb-4bit"
RETRIES = 3


class FlyteTrainerCheckpointCallback(TrainerCallback):
    """Mirror the Trainer's `output_dir` to the Flyte checkpoint after each epoch."""

    def __init__(self, checkpoint: flyte.Checkpoint, output_dir: pathlib.Path) -> None:
        self._checkpoint = checkpoint
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        # Trainer callbacks are synchronous, so use save_sync
        self._checkpoint.save_sync(self._output_dir)


def tiny_instruction_dataset():
    from datasets import Dataset

    rows = [
        "### Instruction:\nSay hello.\n\n### Response:\nHello!",
        "### Instruction:\nWhat is 1+1?\n\n### Response:\n2",
        "### Instruction:\nName a color.\n\n### Response:\nBlue",
    ] * 6
    return Dataset.from_dict({"text": rows})


# {{docs-fragment task}}
@env.task(retries=RETRIES)
def train_unsloth_sft(max_epochs: int = 10) -> float:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    checkpoint = flyte.ctx().checkpoint

    ckpt_dir = pathlib.Path("unsloth_sft")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Restore the previous attempt's checkpoint tree and find the last HF checkpoint.
    hf_resume = None
    prev = checkpoint.load_sync()
    if prev:
        hf_resume = get_last_checkpoint(str(prev))

    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=1,
        report_to="none",
        seed=42,
        dataset_text_field="text",
        max_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=tiny_instruction_dataset(),
        processing_class=tokenizer,
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
        return float(logits[0, 1].mean().item())
# {{/docs-fragment task}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(train_unsloth_sft, max_epochs=10)
    print(run.url)
