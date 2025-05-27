# # Finetuning a Reasoning LLM with Unsloth and Serving with vLLM
#
# In this tutorial, we learn how to create a workflow to finetune a reasoning Qwen3
# large language model with Unsloth and serve it on Union. Unsloth makes finetuning
# LLMs faster and use less memory without degradation in accuracy. Union workflow declarative
# infrastructure makes it easy to specific your computing resources for finetuning.
# Furthermore, we can use Union Serving to serve the finetuned model with ~10 lines of code.

# {{run-on-union}}

# ## Defining Workflow Dependencies
#
# First, we import the modules needed by our workflow:

from union import ImageSpec, FlyteDirectory, task, Resources
from flytekit.extras.accelerators import L4
from flytekit import Cache

from typing import Annotated
from union import Artifact, workflow
from flytekit.extras.accelerators import GPUAccelerator


# Next, we define a `ImageSpec` that contains the python dependencies for the finetuning
# task:

image = ImageSpec(
    name="unsloth-finetune",
    apt_packages=["build-essential"],
    packages=[
        "torch==2.7.0",
        "huggingface-hub[hf_transfer]==0.31.1",
        "pandas==2.2.3",
        "union",
    ],
    registry="ghcr.io/unionai-oss",
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    commands=["uv pip install unsloth==2025.4.7"],
)

# Note that we set `HF_HUB_ENABLE_HF_TRANSFER=1` to use the faster rust-based downloader
# from HuggingFace. `build-essential` is required to use PyTorch compile to optimize the
# model for training.

# ## Finetuning Workflow

# Next we define two artifacts:
# - `qwen-tuned`: The output of Unsloth's finetuning task
# - `vllm-qwen-model`: Convert the Unsloth model into a format that works with VLLM

TUNED_MODEL = Artifact(name="qwen-tuned")
SAVED_VLLM_MODEL = Artifact(name="vllm-qwen-model")

# We define the finetuning by declaring it's resources such as a `L4` GPU and ephemeral storage storage
# used to hold the dataset.


@task(
    container_image=image,
    requests=Resources(mem="23Gi", gpu="1", ephemeral_storage="20Gi", cpu="6"),
    accelerator=L4,
    cache=Cache(version="v1"),
)
def finetune() -> Annotated[FlyteDirectory, TUNED_MODEL]:
    """Finetune model with Unsloth."""
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt
    import pandas as pd
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split="train")

    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append(
                [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            )
        return {
            "conversations": conversations,
        }

    reasoning_conversations = tokenizer.apply_chat_template(
        reasoning_dataset.map(generate_conversation, batched=True)["conversations"],
        tokenize=False,
    )

    dataset = standardize_sharegpt(non_reasoning_dataset)

    non_reasoning_conversations = tokenizer.apply_chat_template(
        dataset["conversations"],
        tokenize=False,
    )

    chat_percentage = 0.75
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations) * (1.0 - chat_percentage)),
        random_state=2407,
    )
    data = pd.concat([pd.Series(reasoning_conversations), pd.Series(non_reasoning_subset)])
    data.name = "text"

    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed=3407)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined_dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    trainer.train()

    lora_path = FlyteDirectory.new("lora_model")
    model.save_pretrained(lora_path.path)
    tokenizer.save_pretrained(lora_path.path)
    return lora_path


# Unsloth can quantize a finetuned model into different formats that are more suitable
# for serving, such as GGUF. In this next task, we convert the model into a 16 bit model
# so that VLLM can easily serve it.


@task(
    container_image=image,
    requests=Resources(mem="23Gi", gpu="1", ephemeral_storage="20Gi", cpu="6"),
    accelerator=L4,
    cache=Cache(version="v1"),
)
def convert_vllm(tuned_model: FlyteDirectory) -> Annotated[FlyteDirectory, SAVED_VLLM_MODEL]:
    """Convert model for VLLM to consume."""
    from unsloth import FastLanguageModel

    tuned_model.download()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=tuned_model.path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    vllm_model_dir = FlyteDirectory.new("lora_model")
    model.save_pretrained_merged(
        vllm_model_dir,
        tokenizer,
        save_method="merged_16bit",
    )

    return vllm_model_dir


# We define a simple workflow that takes the finetuned model and converts it into a format
# for VLLM.
@workflow
def unsloth_finetune():
    tuned_model = finetune()
    convert_vllm(tuned_model=tuned_model)


# ## Defining the VLLM App
#
# Finally, we configure the VLLMApp to serve the finetuned model. The `model` is set
# to the artifact returned by the `convert_vllm` task.
from union.app.llm import VLLMApp

app = VLLMApp(
    name="unsloth-qwen-tuned",
    container_image="ghcr.io/unionai-oss/serving-vllm:0.1.17",
    requests=Resources(mem="23Gi", gpu="1", ephemeral_storage="20Gi", cpu="6"),
    model=SAVED_VLLM_MODEL.query(),
    model_id="unsloth-qwen",
    stream_model=True,
    accelerator=GPUAccelerator("nvidia-l40s"),
)

# To run the finetune workflow with Unsloth:
# ```bash
# union run --remote unsloth_finetune.py unsloth_finetune
# ```
#
# To deploy the VLLM Serving App on Union:
# ```bash
# union deploy apps unsloth_finetune.py unsloth-qwen-tuned
# ```
