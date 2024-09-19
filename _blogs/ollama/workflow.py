import subprocess
import sys
from pathlib import Path

from flytekit import Resources, current_context, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
from flytekitplugins.inference import Model, Ollama

from ollama.utils import (
    PEFTConfig,
    TrainingConfig,
    lora_to_gguf_image,
    image_spec,
    ollama_image,
)

ollama_instance = Ollama(
    model=Model(
        name="phi3-pubmed",
        modelfile='''
FROM phi3:mini-4k 
ADAPTER {inputs.gguf}

TEMPLATE """{{ if .System }}<|system|> 
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"""

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"
PARAMETER num_predict 512
PARAMETER seed 42
PARAMETER temperature 0.05

SYSTEM """
You are a medical research assistant AI that has
been fine-tuned on the latest research. Use the latest knowledge beyond your
initial training data cutoff to provide the most up-to-date information.
"""
''',
    )
)


@task(
    cache=True,
    cache_version="0.1",
    container_image=image_spec,
    requests=Resources(mem="1Gi", cpu="1", ephemeral_storage="8Gi"),
)
def create_dataset(queries: list[str], top_n: int) -> FlyteDirectory:
    from ollama.pubmed_dataset import create_dataset

    working_dir = Path(current_context().working_directory)
    output_dir = working_dir / "dataset"

    create_dataset(
        output_dir,
        queries=queries,
        top_n=top_n,
    )
    return FlyteDirectory(path=str(output_dir))


@task(
    cache=True,
    cache_version="0.4",
    container_image=image_spec,
    accelerator=T4,
    requests=Resources(mem="10Gi", cpu="2", gpu="1"),
    environment={"TOKENIZERS_PARALLELISM": "false"},
)
def phi3_finetune(
    train_args: TrainingConfig, peft_args: PEFTConfig, dataset_dir: FlyteDirectory
) -> tuple[FlyteDirectory, FlyteDirectory]:
    from ollama.train import (
        create_trainer,
        initialize_tokenizer,
        load_model,
        prepare_dataset,
        save_model,
    )

    model = load_model(train_args)
    tokenizer = initialize_tokenizer(train_args.model)
    dataset_splits = prepare_dataset(dataset_dir, train_args, tokenizer)
    trainer = create_trainer(model, train_args, peft_args, dataset_splits, tokenizer)
    save_model(trainer, train_args)

    return FlyteDirectory(train_args.adapter_dir), FlyteDirectory(train_args.output_dir)


@task(
    cache=True,
    cache_version="0.1",
    container_image=lora_to_gguf_image,
    requests=Resources(mem="5Gi", cpu="2", gpu="1"),
    accelerator=T4,
)
def lora_to_gguf(adapter_dir: FlyteDirectory, model_dir: FlyteDirectory) -> FlyteFile:
    adapter_dir.download()
    model_dir.download()
    output_dir = Path(current_context().working_directory)

    subprocess.run(
        [
            sys.executable,
            "/root/llama.cpp/convert_lora_to_gguf.py",
            adapter_dir.path,
            "--base",
            model_dir.path,
            "--outfile",
            str(output_dir / "model.gguf"),
            "--outtype",
            "q8_0",  # quantize the model to 8-bit float representation
        ],
        check=True,
    )

    return FlyteFile(str(output_dir / "model.gguf"))


@task(
    container_image=ollama_image,
    pod_template=ollama_instance.pod_template,
    accelerator=T4,
    requests=Resources(gpu="0"),
)
def model_serving(questions: list[str], gguf: FlyteFile) -> list[str]:
    from openai import OpenAI

    responses = []
    client = OpenAI(
        base_url=f"{ollama_instance.base_url}/v1", api_key="ollama"
    )  # api key required but ignored

    for question in questions:
        completion = client.chat.completions.create(
            model="phi3-pubmed",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical research assistant AI.",
                },
                {"role": "user", "content": question},
            ],
            max_tokens=256,
        )
        responses.append(completion.choices[0].message.content)
    return responses


@workflow
def phi3_ollama(
    train_args: TrainingConfig = TrainingConfig(),
    peft_args: PEFTConfig = PEFTConfig(),
    queries: list[str] = ["crispr therapy", "rna vaccines", "emdr therapy"],
    top_n: int = 3,
    model_queries: list[str] = [
        "What are the most recent clinical trials and outcomes related to CRISPR-based gene therapies for treating genetic disorders?",
        "What are the latest advancements in RNA vaccine development, and how do they compare in efficacy and safety to traditional vaccine platforms?",
    ],
) -> list[str]:
    dataset_dir = create_dataset(queries=queries, top_n=top_n)
    adapter_dir, model_dir = phi3_finetune(
        train_args=train_args, peft_args=peft_args, dataset_dir=dataset_dir
    )
    gguf_file = lora_to_gguf(adapter_dir=adapter_dir, model_dir=model_dir)
    return model_serving(
        questions=model_queries,
        gguf=gguf_file,
    )
