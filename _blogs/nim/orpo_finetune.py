import os
from dataclasses import dataclass, field

import flytekit
import torch
from datasets import load_dataset
from flytekit import ImageSpec, Resources, Secret, task
from flytekit.extras.accelerators import GPUAccelerator
from flytekitplugins.wandb import wandb_init
from huggingface_hub import login
from mashumaro.mixins.json import DataClassJSONMixin
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

from .constants import (
    BUILDER,
    HF_KEY,
    REGISTRY,
    WANDB_ENTITY,
    WANDB_KEY,
    WANDB_PROJECT,
)

orpo_image = ImageSpec(
    name="orpo_llama",
    registry=REGISTRY,
    packages=[
        "transformers==4.41.2",
        "datasets==2.19.2",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "trl==0.9.4",
        "bitsandbytes==0.43.1",
        "flytekitplugins-wandb==1.12.2",
        "flytekit==1.12.0",
        "torch==2.3.1",
        "numpy<2.0.0",
    ],
    cuda="12.2.2",
    cudnn="8",
    python_version="3.12",
    builder=BUILDER,
)


@dataclass
class FineTuningArgs(DataClassJSONMixin):
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model: str = "Samhita/OrpoLlama-3-8B-Instruct"
    dataset_name: str = "mlabonne/orpo-dpo-mix-40k"
    num_samples: int = 1000
    num_train_epochs: int = 3
    learning_rate: float = 8e-6
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    max_prompt_length: int = 512
    lr_scheduler_type: str = "linear"
    optim: str = "paged_adamw_8bit"
    evaluation_strategy: str = "steps"
    eval_steps: float = 0.2
    logging_steps: int = 1
    warmup_steps: int = 10
    report_to: str = "wandb"
    output_dir: str = "./results/"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ]
    )
    lora_task_type: str = "CAUSAL_LM"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    device_map: str = "auto"


wandb_secret = Secret(key=WANDB_KEY)


@task(
    cache=True,
    cache_version="0.1",
    requests=Resources(mem="25Gi", cpu="5", gpu="1"),
    secret_requests=[
        Secret(key=HF_KEY),
        wandb_secret,
    ],
    container_image=orpo_image,
    accelerator=GPUAccelerator("nvidia-tesla-l4"),
    environment={"CUDA_LAUNCH_BLOCKING": "1"},
)
@wandb_init(project=WANDB_PROJECT, entity=WANDB_ENTITY, secret=wandb_secret)
def llama_8b_instruct_finetune(args: FineTuningArgs) -> str:
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=args.lora_task_type,
        target_modules=args.lora_target_modules,
    )

    login(token=flytekit.current_context().secrets.get(HF_KEY))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=args.device_map,
        attn_implementation=attn_implementation,
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="all")
    dataset = dataset.shuffle(seed=42).select(range(args.num_samples))

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(format_chat_template, num_proc=os.cpu_count())
    dataset = dataset.train_test_split(test_size=0.01)

    orpo_args = ORPOConfig(
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=0.1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        report_to=args.report_to,
        output_dir=args.output_dir,
        hub_model_id=args.new_model,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.push_to_hub()

    return args.new_model
