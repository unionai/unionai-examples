# # Benchmarking the Efficiency of LLM Finetuning with the Liger Kernel
#
# This tutorial demonstrates how to run a fine-tuning benchmarking experiment
# using the Liger kernel. We'll use the HuggingFace `transformers` library and the
# `alpaca` dataset to finetune a pre-trained Llama 3 8B model with and without the
# Liger kernel and compare the results.

import itertools
import json
import os
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import pandas as pd
from flytekit import (
    current_context,
    dynamic,
    map_task,
    task,
    workflow,
    Deck,
    ImageSpec,
    Resources,
    Secret,
)
from flytekit.extras.accelerators import A100
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.deck.renderer import TableRenderer, FrameProfilingRenderer
from flytekitplugins.wandb import wandb_init

import transformers
from callback import EfficiencyCallback


image = ImageSpec(
    name="liger-kernel-finetuning",
    packages=[
        "flytekitplugins-deck-standard",
        "flytekitplugins-wandb",
        "datasets==2.21.0",
        "pandas==2.2.2",
        "matplotlib==3.9.2",
        "huggingface-hub==0.24.6",
        "transformers==4.43.3",
        "trl==0.10.1",
        "torch==2.4.1",
        "liger-kernel==0.2.1",
    ],
    apt_packages=["build-essential"],
    cuda="12.1",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)

WANDB_PROJECT = "liger-kernel-finetuning"
WANDB_ENTITY = "niels-bantilan"


@dataclass
class CustomArguments:
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    dataset_name: str = "tatsu-lab/alpaca"
    max_seq_length: int = 512
    use_liger: bool = False


@dataclass
class ExperimentArguments:
    use_liger: list[bool] = field(default_factory=lambda: [True])
    per_device_train_batch_size: list[int] = field(default_factory=lambda: [8])


@dataclass
class TrainingArguments(CustomArguments):
    bf16: bool = True
    max_steps: int = 10
    num_train_epochs: int = 1
    optim: str = "adamw_torch"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    eval_strategy: str = "no"
    save_strategy: str = "no"
    learning_rate: float = 0.000006
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 1
    include_num_input_tokens_seen: bool = True
    report_to: str = "none"
    seed: int = 42
    fsdp: str = ""
    fsdp_config: dict | None = field(default=None)


@dataclass
class TrainingResult:
    model_dir: FlyteDirectory
    training_history: FlyteFile
    training_args: TrainingArguments


WANDB_SECRET = Secret(key="wandb_api_key")


@task(
    container_image=image,
    limits=Resources(mem="24Gi", cpu="12", gpu="1"),
    accelerator=A100,
    cache=True,
    cache_version="v11",
    secret_requests=[WANDB_SECRET],
    environment={"TOKENIZERS_PARALLELISM": "false"},
)
@wandb_init(project=WANDB_PROJECT, entity=WANDB_ENTITY, secret=WANDB_SECRET)
def train_model(training_args: TrainingArguments) -> TrainingResult:

    import torch
    from datasets import load_dataset
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    train_dir = working_dir / "models"

    parser = transformers.HfArgumentParser((transformers.TrainingArguments, CustomArguments))
    hf_training_args, custom_args = parser.parse_dict(
        {"output_dir": train_dir, **asdict(training_args)}
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.model_name,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(training_args.dataset_name)["train"].train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # need to provide the response_prompt with enough context for the tokenizer to encode
    # the repsonse prompt ids correctly:
    # https://huggingface.co/docs/trl/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    response_prompt = tokenizer.encode("\n### Response:", add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_prompt,
        pad_to_multiple_of=16,
    )

    if custom_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            training_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name,
            trust_remote_code=True,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
    print(f"Model:\n{model}")
    print(f"Training arguments:\n{hf_training_args}")

    def formatting_prompts_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        args=hf_training_args,
        data_collator=collator,
        tokenizer=tokenizer,
        max_seq_length=custom_args.max_seq_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        callbacks=[EfficiencyCallback()],
    )
    trainer.train()

    # save training history to a file
    training_history_file = working_dir / "training_history.json"
    with training_history_file.open("w") as f:
        json.dump(trainer.state.log_history, f)

    return TrainingResult(
        FlyteDirectory(hf_training_args.output_dir),
        FlyteFile(training_history_file),
        training_args,
    )


@task(
    container_image=image,
    cache=True,
    cache_version="v1",
)
def prepare_experiment_args(
    experiment_args: ExperimentArguments,
    training_args: TrainingArguments,
) -> list[TrainingArguments]:
    training_args_list = []
    for use_liger, bs in itertools.product(
        *[experiment_args.use_liger, experiment_args.per_device_train_batch_size],
    ):
        args = deepcopy(training_args)
        args.use_liger = use_liger
        args.per_device_train_batch_size = bs
        training_args_list.append(args)
    return training_args_list


@task(
    container_image=image,
    enable_deck=True,
    limits=Resources(mem="8Gi", cpu="4"),
)
def analyze_results(results: list[Optional[TrainingResult]]) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    ctx = current_context()
    history_columns = ["epoch", "step"]
    metrics = [
        "step_tokens_per_second",
        "avg_tokens_per_second",
        "step_peak_memory_allocated_MB",
        "step_peak_memory_reserved_MB",
        "total_peak_memory_allocated_MB",
        "total_peak_memory_reserved_MB",
    ]
    experiment_vars = [*ExperimentArguments.__dataclass_fields__]

    dataframe = []
    for result in results:
        if result is None:
            continue
        result.training_history.download()
        exp_vars = {k: getattr(result.training_args, k) for k in experiment_vars}
        with open(result.training_history.path, "r") as f:
            dataframe.append(pd.read_json(f).assign(**exp_vars))

    dataframe = pd.concat(dataframe)
    analysis_df = (
        dataframe[history_columns + experiment_vars + metrics].dropna().drop_duplicates()
    )
    # try converting experiment_vars to numeric types
    for col in experiment_vars:
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors="ignore", downcast="integer")

    grpby = analysis_df.groupby(experiment_vars)
    avg_tokens_per_second = grpby.avg_tokens_per_second.last().to_frame()
    step_peak_memory_reserved_mb = grpby.step_peak_memory_reserved_MB.max().to_frame()

    benchmark_deck = Deck("Benchmarking Results")
    fig, ax = plt.subplots(1, 2)
    avg_tokens_per_second.plot.barh(ax=ax[0])
    step_peak_memory_reserved_mb.plot.barh(ax=ax[1])
    benchmark_deck.append(_convert_fig_into_html(fig))
    benchmark_deck.append(TableRenderer().to_html(df=avg_tokens_per_second.reset_index()))
    benchmark_deck.append(
        TableRenderer().to_html(df=step_peak_memory_reserved_mb.reset_index())
    )

    benchmark_data_deck = Deck("Benchmarking Data")
    benchmark_data_deck.append(FrameProfilingRenderer().to_html(df=analysis_df))

    ctx.decks.insert(0, benchmark_data_deck)
    ctx.decks.insert(0, benchmark_deck)

    return analysis_df


@workflow
def training_workflow(
    experiment_args: ExperimentArguments = ExperimentArguments(),
    training_args: TrainingArguments = TrainingArguments(),
) -> TrainingResult:
    return train_model(training_args=training_args)


@workflow
def benchmarking_experiment(
    experiment_args: ExperimentArguments = ExperimentArguments(),
    training_args: TrainingArguments = TrainingArguments(),
) -> tuple[list[TrainingResult], pd.DataFrame]:
    training_args_list = prepare_experiment_args(experiment_args, training_args)
    results = map_task(train_model, min_success_ratio=0.1)(training_args=training_args_list)
    analysis = analyze_results(results=results)
    return results, analysis


# Helper function to convert a matplotlib figure into an HTML string
def _convert_fig_into_html(fig) -> str:
    import io
    import base64
    import matplotlib as mpl

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'
