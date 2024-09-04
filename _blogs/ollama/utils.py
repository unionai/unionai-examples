from dataclasses import dataclass, field
from typing import Any, Optional

from flytekit import ImageSpec
from mashumaro.mixins.json import DataClassJSONMixin

from ollama.constants import REGISTRY

image_spec = ImageSpec(
    name="phi3-finetune",
    apt_packages=["git", "build-essential"],
    requirements="requirements.txt",
    registry=REGISTRY,
)

hf_to_gguf_image = ImageSpec(
    name="gguf-ollama",
    apt_packages=["git"],
    registry=REGISTRY,
    packages=["huggingface_hub"],
    python_version="3.11",
).with_commands(
    [
        "git clone --branch b3046 https://github.com/ggerganov/llama.cpp /root/llama.cpp",
        "pip install -r /root/llama.cpp/requirements.txt",
    ]
)

ollama_image = ImageSpec(
    name="phi3-ollama-serve",
    registry=REGISTRY,
    apt_packages=["git"],
    packages=[
        "git+https://github.com/flyteorg/flytekit.git@bcc13f799da3ce28e81d6060fc5776b6b4bca0a0#subdirectory=plugins/flytekit-inference"
    ],
)


@dataclass
class TrainingConfig(DataClassJSONMixin):
    model: str = "microsoft/Phi-3-mini-4k-instruct"
    bf16: bool = True
    do_eval: bool = False
    learning_rate: float = 5.0e-06
    log_level: str = "info"
    logging_steps: int = 20
    logging_strategy: str = "steps"
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 1
    max_steps: int = -1
    output_dir: str = "./checkpoint_dir"
    overwrite_output_dir: bool = True
    per_device_eval_batch_size: int = 4
    per_device_train_batch_size: int = 4
    remove_unused_columns: bool = True
    save_steps: int = 100
    save_total_limit: int = 1
    seed: int = 0
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.2
    dataloader_num_proc: int = 1
    model_max_length: int = 1024
    test_size: float = 0.01
    adapter_dir: str = "./adapter"
    output_dir: str = "./merged_adapters"


@dataclass
class PEFTConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: str = "all-linear"
    modules_to_save: Optional[Any] = None
