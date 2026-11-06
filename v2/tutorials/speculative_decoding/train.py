"""Train a draft model (EAGLE-3, DFlash, or DSpark) with speculators, in online mode.

All three drafters condition on the target model's hidden states. Storing them offline costs about
32 KB per token for an 8B target, which is terabytes for a modest dataset. In online mode,
a vLLM server on two of the task's GPUs computes hidden states on demand and the trainer on
the other two consumes them, so nothing but the checkpoint is ever stored.

The task's report charts loss and per-position draft accuracy live as training runs.
"""

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import flyte
from flyte.io import Dir, File

import train_report
from config import TARGET_MODEL, TRAIN_NUM_GPUS, train_env
from recipes import RECIPES
from vllm_server import TOOL_ARGS, start_server

SEQ_LEN = 4096
HIDDEN_STATES = "/dev/shm/hidden_states"
# GPU layout: which devices serve hidden states and which train.
if TRAIN_NUM_GPUS == 4:
    VLLM_GPUS, TRAIN_GPUS, VLLM_REPLICAS, TRAINERS = "0,1", "2,3", 2, 2
else:
    VLLM_GPUS, TRAIN_GPUS, VLLM_REPLICAS, TRAINERS = "0", "0", 1, 1


def _best_checkpoint(save_path: Path) -> Path:
    """`checkpoint_best` when validation picked one, else the last epoch."""
    best = save_path / "checkpoint_best"
    if best.exists():
        return best.resolve()
    epochs = sorted((p for p in save_path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    return epochs[-1]


def _for_prepare_data(path: str) -> str:
    """speculators reads raw conversations from a `conversations` key; traces use `messages`."""
    out = path + ".prepared.jsonl"
    with open(path) as src, open(out, "w") as dst:
        for line in src:
            row = json.loads(line)
            if "input_ids" not in row and "messages" in row:
                row["conversations"] = row.pop("messages")
            dst.write(json.dumps(row) + "\n")
    return out


def _is_tokenized(path: str) -> bool:
    with open(path) as f:
        return "input_ids" in json.loads(f.readline())


# {{docs-fragment prepare}}
def prepare_data(paths: list[str], render_endpoint: str, out: str) -> None:
    """Tokenize training files with the target's chat template into one dataset.

    Regenerated chat rows arrive tokenized; agent traces are rendered through the vLLM
    server. speculators 0.9 cannot concatenate the two kinds in one call (their token id
    types differ), so each kind is prepared on its own and the results merged.
    """
    from datasets import concatenate_datasets, load_from_disk
    from speculators.train.vocab_mapping import save_token_frequency_distribution

    groups = [g for g in ([p for p in paths if _is_tokenized(p)], [p for p in paths if not _is_tokenized(p)]) if g]
    parts = []
    for i, group in enumerate(groups):
        part = f"{out}-{i}"
        subprocess.run(
            [
                "speculators", "prepare-data",
                "--model", TARGET_MODEL,
                *[arg for path in group for arg in ("--data", path)],
                "--output", part,
                "--seq-length", str(SEQ_LEN),
                "--render-endpoint", render_endpoint,
            ],
            check=True,
        )
        parts.append(load_from_disk(part))
    # Cast every part to 64-bit token ids, then merge and shuffle.
    target = next((p.features for p in parts if "int64" in str(p.features["input_ids"])), parts[0].features)
    merged = concatenate_datasets([p.cast(target) for p in parts]).shuffle(seed=0)
    merged.save_to_disk(out)
    # Token frequencies pick the reduced draft vocabulary for drafters trained from scratch.
    save_token_frequency_distribution(merged, f"{out}/token_freq.pt")
# {{/docs-fragment prepare}}


# Flags that shape a new draft model. speculators rejects them with --from-pretrained.
ARCHITECTURE_FLAGS = {"--num-layers", "--draft-vocab-size"}


def _without(args: list[str], flags: set[str]) -> list[str]:
    """Drop `flag value` pairs for the given flags."""
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg in flags:
            skip = True
        else:
            out.append(arg)
    return out


def _check_prepared(inputs: list[str], prepared: str, min_kept: float = 0.9) -> None:
    """prepare-data skips rows it cannot render. Fail instead of training on a fraction."""
    from datasets import load_from_disk

    raw = 0
    for path in inputs:
        with open(path) as f:
            raw += sum(1 for _ in f)
    kept = len(load_from_disk(prepared))
    print(f"prepare-data kept {kept} of {raw} rows")
    if kept < min_kept * raw:
        raise RuntimeError(f"prepare-data kept only {kept} of {raw} rows; check its errors above")


# {{docs-fragment stream}}
async def run_and_chart(cmd: list[str], env: dict, history: train_report.History) -> None:
    """Run the trainer, parse its metric lines, and refresh the report as it trains."""
    process = await asyncio.create_subprocess_exec(
        *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    last_flush = 0.0
    async for raw in process.stdout:
        line = raw.decode(errors="replace").rstrip()
        print(line, flush=True)
        if history.parse(line) and time.monotonic() - last_flush > 20:
            await flyte.report.replace.aio(train_report.render(history))
            await flyte.report.flush.aio()
            last_flush = time.monotonic()
    if await process.wait() != 0:
        raise RuntimeError(f"training failed with exit code {process.returncode}")
# {{/docs-fragment stream}}


# {{docs-fragment train}}
@train_env.task(cache="auto", retries=1, report=True)
async def train_drafter(
    data: list[File],
    method: str = "eagle3",
    init_from: str | None = None,
    warm_start: Dir | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    max_steps: int | None = None,
) -> Dir:
    """Fine-tune (or train from scratch) a draft model on target-generated data.

    `method` picks a recipe: "eagle3", "dflash", or "dspark". `warm_start` continues from a
    previous run's head. Otherwise `init_from` overrides the recipe's Hugging Face head; an
    empty string trains from scratch. Unset arguments come from the recipe.

    The task is cached: the same data and arguments return the drafter already trained, so
    the benchmark reuses what each family's pipeline trained.
    """
    recipe = RECIPES[method]
    init_from = recipe.init_from if init_from is None else init_from
    lr = recipe.lr if lr is None else lr
    epochs = recipe.epochs if epochs is None else epochs
    shared_gpu = VLLM_GPUS == TRAIN_GPUS

    # Hidden-state server: vLLM capturing exactly the layers this drafter reads.
    server = start_server(
        gpus=VLLM_GPUS,
        hidden_states_path=HIDDEN_STATES,
        target_layers=recipe.target_layers,
        extra_args=[
            # Rendering conversations that carry tool schemas needs the tool parser.
            *TOOL_ARGS,
            "--data-parallel-size", str(VLLM_REPLICAS),
            "--max-model-len", str(SEQ_LEN + 512),
            "--gpu-memory-utilization", str(recipe.vllm_memory if shared_gpu else 0.85),
        ],
    )
    start = "previous drafter" if warm_start else (init_from or "scratch")
    history = train_report.History(
        method=method,
        config={
            "method": method,
            "start from": start,
            "data files": str(len(data)),
            "epochs": str(epochs),
            "max steps": str(max_steps or "all"),
            "learning rate": str(lr),
            "target layers": ", ".join(map(str, recipe.target_layers)),
            "method flags": " ".join(recipe.train_args),
            "sequence length": str(SEQ_LEN),
            "GPUs": f"vLLM {VLLM_GPUS}, trainer {TRAIN_GPUS}",
        },
    )
    try:
        # Tokenize every file with the target's chat template. Regenerated chat files
        # are already tokenized; agent traces are rendered through the server.
        local = [_for_prepare_data(await f.download()) for f in data]
        prepare_data(local, server.base_url, "/tmp/prepared")
        _check_prepared(local, "/tmp/prepared")

        method_args = list(recipe.train_args)
        if warm_start is not None or init_from:
            # A pretrained head fixes its own architecture and vocabulary.
            method_args = _without(method_args, ARCHITECTURE_FLAGS)
            source = await warm_start.download() if warm_start is not None else init_from
            draft_args = ["--from-pretrained", source]
        else:
            draft_args = []

        save_path = Path("/tmp/checkpoints")
        await run_and_chart(
            [
                "torchrun", "--standalone", "--nproc_per_node", str(TRAINERS),
                "-m", "speculators.train",
                "--speculator-type", method,
                "--verifier-name-or-path", TARGET_MODEL,
                *draft_args,
                "--target-layer-ids", *map(str, recipe.target_layers),
                *method_args,
                "--data-path", "/tmp/prepared",
                "--vllm-endpoint", f"{server.base_url}/v1",
                "--hidden-states-path", HIDDEN_STATES,
                "--on-missing", "generate",
                "--save-path", str(save_path),
                "--save-best",
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--total-seq-len", str(SEQ_LEN),
                "--num-workers", "4",
                *(["--max-steps", str(max_steps)] if max_steps else []),
            ],
            # A wide console keeps each metric line on one line for the parser.
            env={**os.environ, "CUDA_VISIBLE_DEVICES": TRAIN_GPUS, "COLUMNS": "100000"},
            history=history,
        )
    finally:
        server.stop()

    # Upload only what vLLM needs to serve the head: config and weights.
    best = _best_checkpoint(save_path)
    out = Path("/tmp/drafter")
    out.mkdir()
    for f in best.iterdir():
        if f.suffix in {".json", ".safetensors", ".py"}:
            shutil.copy(f, out / f.name)
    history.checkpoint = best.name
    await flyte.report.replace.aio(train_report.render(history, done=True))
    await flyte.report.flush.aio()
    return await Dir.from_local(str(out))
# {{/docs-fragment train}}
