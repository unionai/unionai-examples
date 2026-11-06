"""Train an EAGLE-3 draft head with speculators, in online mode.

EAGLE-3 conditions on the target model's hidden states. Storing them offline costs about
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
from config import PUBLIC_DRAFTER, TARGET_MODEL, TRAIN_NUM_GPUS, train_env
from vllm_server import start_server

SEQ_LEN = 4096
HIDDEN_STATES = "/dev/shm/hidden_states"
# GPU layout: which devices serve hidden states, which train, and vLLM's memory share.
if TRAIN_NUM_GPUS == 4:
    VLLM_GPUS, TRAIN_GPUS, VLLM_REPLICAS, TRAINERS, VLLM_MEMORY = "0,1", "2,3", 2, 2, 0.85
else:
    VLLM_GPUS, TRAIN_GPUS, VLLM_REPLICAS, TRAINERS, VLLM_MEMORY = "0", "0", 1, 1, 0.45


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
@train_env.task(retries=1, report=True)
async def train_drafter(
    data: list[File],
    init_from: str = PUBLIC_DRAFTER,
    warm_start: Dir | None = None,
    epochs: int = 2,
    lr: float = 5e-5,
    ttt_steps: int = 3,
    max_steps: int | None = None,
) -> Dir:
    """Fine-tune (or train from scratch) an EAGLE-3 head on target-generated data.

    `warm_start` continues from a previous run's draft head. Without it, `init_from`
    names a Hugging Face draft head to fine-tune; an empty string trains from scratch.
    """
    # Hidden-state server: vLLM with speculators' extraction connector.
    server = start_server(
        gpus=VLLM_GPUS,
        hidden_states_path=HIDDEN_STATES,
        extra_args=[
            "--data-parallel-size", str(VLLM_REPLICAS),
            "--max-model-len", str(SEQ_LEN + 512),
            "--gpu-memory-utilization", str(VLLM_MEMORY),
        ],
    )
    history = train_report.History(
        config={
            "start from": "previous drafter" if warm_start else (init_from or "scratch"),
            "data files": str(len(data)),
            "epochs": str(epochs),
            "max steps": str(max_steps or "all"),
            "learning rate": str(lr),
            "TTT steps": str(ttt_steps),
            "sequence length": str(SEQ_LEN),
            "GPUs": f"vLLM {VLLM_GPUS}, trainer {TRAIN_GPUS}",
        }
    )
    try:
        # Tokenize every file with the target's chat template. Regenerated chat files
        # are already tokenized; agent traces are rendered through the server.
        local = [_for_prepare_data(await f.download()) for f in data]
        data_args = [arg for path in local for arg in ("--data", path)]
        subprocess.run(
            [
                "speculators", "prepare-data",
                "--model", TARGET_MODEL,
                *data_args,
                "--output", "/tmp/prepared",
                "--seq-length", str(SEQ_LEN),
                "--render-endpoint", server.base_url,
                "--no-skip-token-freq",
            ],
            check=True,
        )

        _check_prepared(local, "/tmp/prepared")

        if warm_start is not None:
            draft_args = ["--from-pretrained", await warm_start.download()]
        elif init_from:
            draft_args = ["--from-pretrained", init_from]
        else:
            draft_args = ["--draft-vocab-size", "32000"]

        save_path = Path("/tmp/checkpoints")
        await run_and_chart(
            [
                "torchrun", "--standalone", "--nproc_per_node", str(TRAINERS),
                "-m", "speculators.train",
                "--verifier-name-or-path", TARGET_MODEL,
                *draft_args,
                "--data-path", "/tmp/prepared",
                "--vllm-endpoint", f"{server.base_url}/v1",
                "--hidden-states-path", HIDDEN_STATES,
                "--on-missing", "generate",
                "--save-path", str(save_path),
                "--save-best",
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--ttt-steps", str(ttt_steps),
                "--total-seq-len", str(SEQ_LEN),
                "--num-workers", "4",
                *(["--max-steps", str(max_steps)] if max_steps else []),
            ],
            # A wide console keeps each metric line on one line for the parser.
            env={**os.environ, "CUDA_VISIBLE_DEVICES": TRAIN_GPUS, "COLUMNS": "1000"},
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
