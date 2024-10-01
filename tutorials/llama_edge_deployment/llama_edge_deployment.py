import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from flytekit import LaunchPlan, Resources, Secret, current_context, task, workflow
from flytekit.core.artifact import Inputs
from flytekit.extras.accelerators import A100, L4
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.wandb import wandb_init
from union.artifacts import OnArtifact

from .utils import (
    ModelArtifact,
    download_artifacts_image,
    llm_mlc_image,
    model_training_image,
    modelcard_image,
)

WANDB_SECRET = Secret(key="wandb-api-key")
WANDB_PROJECT = "<WANDB_PROJECT>"
WANDB_ENTITY = "<WANDB_ENTITY>"


@dataclass
class TrainingArguments:
    output_dir: str
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_8bit"
    num_train_epochs: int = 1
    evaluation_strategy: str = "steps"
    eval_steps: float = 0.2
    logging_steps: int = 1
    warmup_steps: int = 10
    logging_strategy: str = "steps"
    learning_rate: float = 8e-6
    fp16: bool = True
    bf16: bool = False
    group_by_length: bool = True
    report_to: str = "wandb"


@task(
    cache=True,
    cache_version="0.1",
    container_image=download_artifacts_image,
)
def download_dataset(dataset: str, language: str) -> FlyteDirectory:
    from datasets import load_dataset

    working_dir = Path(current_context().working_directory)
    cached_dataset_dir = working_dir / "cached_dataset"
    load_dataset(dataset, language, cache_dir=cached_dataset_dir)

    return cached_dataset_dir


@task(
    cache=True,
    cache_version="0.1",
    requests=Resources(mem="10Gi"),
    secret_requests=[Secret(key="hf-api-key")],
    container_image=download_artifacts_image,
)
def download_model(model_name: str) -> FlyteDirectory:
    from huggingface_hub import login, snapshot_download

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    cached_model_dir = working_dir / "cached_model"

    login(token=ctx.secrets.get(key="hf-api-key"))
    snapshot_download(model_name, local_dir=cached_model_dir)
    return cached_model_dir


@task(
    cache=True,
    cache_version="0.2",
    container_image=model_training_image,
    accelerator=A100,
    requests=Resources(mem="20Gi", gpu="1", cpu="5"),
    secret_requests=[WANDB_SECRET],
    environment={"TOKENIZERS_PARALLELISM": "false"},
)
@wandb_init(project=WANDB_PROJECT, entity=WANDB_ENTITY, secret=WANDB_SECRET)
def train_model(
    train_args: TrainingArguments,
    dataset_dir: FlyteDirectory,
    model_dir: FlyteDirectory,
) -> FlyteDirectory:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        HfArgumentParser,
        TrainingArguments,
    )
    from trl import SFTTrainer, setup_chat_format

    dataset_dir.download()
    model_dir.download()

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir.path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir.path)
    model, tokenizer = setup_chat_format(model, tokenizer)

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    # Importing the dataset
    dataset = load_dataset(dataset_dir.path)

    # Select train and test datasets from the loaded split
    train_dataset = dataset["train"]
    test_dataset = dataset.get("test", None)  # In case there's no test split

    # Shuffle and select a subset (optional)
    train_dataset = train_dataset.shuffle(seed=65).select(range(1000))
    test_dataset = test_dataset.shuffle(seed=65).select(range(1000))

    def format_chat_template(row):
        row_json = [
            {"role": "user", "content": row["inputs"]},
            {"role": "assistant", "content": row["targets"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    # Apply the format_chat_template function in parallel (num_proc=4)
    train_dataset = train_dataset.map(
        format_chat_template,
        num_proc=4,
    )

    # Apply format_chat_template to the test dataset if it exists
    if test_dataset is not None:
        test_dataset = test_dataset.map(
            format_chat_template,
            num_proc=4,
        )

    parser = HfArgumentParser(TrainingArguments)
    hf_training_args = parser.parse_dict(asdict(train_args))[0]

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=hf_training_args,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(train_args.output_dir)

    return FlyteDirectory(train_args.output_dir)


@task(
    cache=True,
    cache_version="0.2",
    container_image=model_training_image,
    requests=Resources(mem="20Gi", gpu="1"),
    accelerator=L4,
)
def merge_model(
    model_name: str,
    dataset: str,
    model_dir: FlyteDirectory,
    adapter_dir: FlyteDirectory,
) -> Annotated[
    FlyteDirectory, ModelArtifact(model=Inputs.model_name, dataset=Inputs.dataset)
]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import setup_chat_format

    model_dir.download()
    adapter_dir.download()

    # Reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir.path)
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        model_dir.path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model_reload, adapter_dir.path)

    model = model.merge_and_unload()

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    merged_model = working_dir / "merged_model"

    model.save_pretrained(merged_model)
    tokenizer.save_pretrained(merged_model)

    return FlyteDirectory(merged_model)


@task(
    cache=True,
    cache_version="0.1",
    requests=Resources(mem="20Gi", gpu="1"),
    accelerator=L4,
    container_image=llm_mlc_image,
)
def convert_model_weights_to_mlc(
    merged_model_dir: FlyteDirectory, conversion_template: str, quantization: str
) -> FlyteDirectory:
    merged_model_dir.download()
    output_dir = Path(current_context().working_directory)

    subprocess.run(
        [
            "python",
            "-m",
            "mlc_llm",
            "convert_weight",
            merged_model_dir.path,
            "--quantization",
            quantization,
            "-o",
            str(output_dir / "finetuned-model-MLC"),
        ],
        check=True,
    )

    subprocess.run(
        [
            "python",
            "-m",
            "mlc_llm",
            "gen_config",
            merged_model_dir.path,
            "--quantization",
            quantization,
            "--conv-template",
            conversion_template,
            "-o",
            str(output_dir / "finetuned-model-MLC"),
        ],
        check=True,
    )
    return FlyteDirectory(str(output_dir / "finetuned-model-MLC"))


@task(
    cache=True,
    cache_version="0.1",
    container_image=modelcard_image,
    requests=Resources(mem="10Gi", cpu="2"),
    secret_requests=[Secret(key="hf-api-key")],
)
def push_to_hf(model_directory: FlyteDirectory, hf_repo_id: str) -> str:
    from huggingface_hub import create_repo, repo_exists, upload_folder

    hf_token = current_context().secrets.get(key="hf-api-key")

    # Check if the repository already exists
    if not repo_exists(hf_repo_id):
        create_repo(
            hf_repo_id,
            private=False,
            token=hf_token,
            repo_type="model",
        )
    else:
        print(f"Repository '{hf_repo_id}' already exists.")

    model_directory.download()
    upload_folder(
        repo_id=hf_repo_id,
        folder_path=model_directory.path,
        path_in_repo=None,
        commit_message="Upload MLC weights",
        token=hf_token,
    )

    model_url = f"https://huggingface.co/{hf_repo_id}"
    return model_url


@workflow
def finetuning_wf(
    dataset: str = "CohereForAI/aya_collection_language_split",
    language: str = "telugu",
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    train_args: TrainingArguments = TrainingArguments(
        output_dir="finetuned-llama-3-8b"
    ),
) -> FlyteDirectory:
    dataset_dir = download_dataset(dataset=dataset, language=language)
    model_dir = download_model(model_name=model_name)
    adapter_dir = train_model(
        train_args=train_args, dataset_dir=dataset_dir, model_dir=model_dir
    )
    return merge_model(
        model_name=model_name,
        dataset=dataset,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
    )


@workflow
def convert_to_mlc_wf(
    conversion_template: str = "llama-3",  # ref: https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_llm/conversation_template
    quantization: str = "q4f16_1",  # quantize the model to 4-bit float representation
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset: str = "CohereForAI/aya_collection_language_split",
    merged_model_dir: FlyteDirectory = ModelArtifact.query(
        model=Inputs.model, dataset=Inputs.dataset
    ),
    hf_repo_id: str = "<HF_REPO_ID>",
) -> str:
    mlc_weights = convert_model_weights_to_mlc(
        merged_model_dir=merged_model_dir,
        conversion_template=conversion_template,
        quantization=quantization,
    )
    return push_to_hf(model_directory=mlc_weights, hf_repo_id=hf_repo_id)


LaunchPlan.create(
    "finetuning_completion_trigger",
    convert_to_mlc_wf,
    trigger=OnArtifact(trigger_on=ModelArtifact),
)
