# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b35",
#    "torch>=2.0",
#    "transformers>=4.41",
#    "accelerate",
#    "async-lru",
#    "huggingface-hub>=0.24",
#    "hf-transfer",
# ]
# main = "batch_llm_pipeline"
# params = "jsonl_path='prompts.jsonl', max_new_tokens=128"
# ///

"""
Batch LLM Inference Pipeline
=============================

Demonstrates batch inference with a small LLM (Qwen2.5-0.5B) using
``InferencePipeline`` from ``flyte.extras``.

Architecture::

    [I/O: Read JSONL lines]           Async file read
            |
    [CPU: Tokenize + estimate tokens] preprocess_executor (4 threads)
            |
    [GPU: model.generate()]           DynamicBatcher with token budgeting, gpu_pool (1 thread)
            |
    [CPU: Decode + simple eval]       Event loop (lightweight)

The preprocessing tokenizes each prompt and estimates its token count so
the DynamicBatcher can assemble token-budgeted GPU batches. This prevents
OOM from batches with too many total tokens while still filling each batch
as much as possible.

Usage::

    flyte run batch_llm_pipeline.py batch_generate
"""

import asyncio
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from async_lru import alru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

import flyte
import flyte.io
from flyte.extras import InferencePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pools
# ---------------------------------------------------------------------------

_cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu")
_gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_INPUT_TOKENS = 512   # truncate inputs longer than this
TARGET_BATCH_TOKENS = 4096  # token budget per GPU batch

# ---------------------------------------------------------------------------
# Image & environments
# ---------------------------------------------------------------------------

image = flyte.Image.from_uv_script(
    __file__, name="batch_llm_pipeline_image"
).with_pip_packages("unionai-reuse>=0.1.9")

worker = flyte.TaskEnvironment(
    name="llm_pipeline_worker",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    reusable=flyte.ReusePolicy(
        replicas=2,
        concurrency=4,  # 4 concurrent task streams per replica
        idle_ttl=120,
        scaledown_ttl=120,
    ),
)

driver = flyte.TaskEnvironment(
    name="llm_pipeline_driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[worker],
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PromptItem:
    """A single JSONL line with a prompt and optional expected answer."""
    prompt: str
    expected: str  # expected substring for simple eval (empty if none)
    line_idx: int
    num_tokens: int = 0  # populated after tokenization


@dataclass
class TokenizedPrompt:
    """Tokenized prompt ready for GPU inference."""
    input_ids: torch.Tensor      # [seq_len]
    attention_mask: torch.Tensor  # [seq_len]
    num_tokens: int

    def estimate_cost(self) -> int:
        """Token count as cost — DynamicBatcher uses this for budgeting."""
        return self.num_tokens


@dataclass
class GenerationResult:
    """Final output after postprocessing."""
    line_idx: int
    prompt: str
    response: str
    num_input_tokens: int
    num_output_tokens: int
    expected: str
    match: bool | None  # None if no expected answer


# ---------------------------------------------------------------------------
# Model loading (process-level singleton)
# ---------------------------------------------------------------------------


@alru_cache(maxsize=1)
async def _load_model_and_tokenizer():
    loop = asyncio.get_running_loop()

    def _load():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        model.eval()

        # Warmup at realistic batch size + generation length to pre-allocate KV cache
        if torch.cuda.is_available():
            dummy = tokenizer(["warmup " * 20] * 16, return_tensors="pt", padding=True)
            dummy = {k: v.to(model.device) for k, v in dummy.items()}
            with torch.no_grad():
                model.generate(**dummy, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)

        logger.warning("Model %s loaded on %s", MODEL_NAME, model.device)
        return model, tokenizer

    return await loop.run_in_executor(_gpu_pool, _load)


# ---------------------------------------------------------------------------
# Pipeline stage functions
# ---------------------------------------------------------------------------


async def preprocess(item: PromptItem) -> TokenizedPrompt:
    """Tokenize a prompt on the CPU threadpool.

    Truncates to MAX_INPUT_TOKENS and returns the token count for
    cost-based batch budgeting.
    """
    _, tokenizer = await _load_model_and_tokenizer()
    loop = asyncio.get_running_loop()

    def _tokenize():
        encoded = tokenizer(
            item.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        num_tokens = encoded["input_ids"].shape[1]
        # Store actual token count on the item for postprocessing
        item.num_tokens = num_tokens
        return TokenizedPrompt(
            input_ids=encoded["input_ids"].squeeze(0),
            attention_mask=encoded["attention_mask"].squeeze(0),
            num_tokens=num_tokens,
        )

    return await loop.run_in_executor(_cpu_pool, _tokenize)


async def inference_batch(
    batch: list[TokenizedPrompt],
    max_new_tokens: int = 128,
) -> list[str]:
    """Run model.generate() on a batch of tokenized prompts.

    Pads the batch to uniform length, generates on GPU, decodes back
    to text. Returns only the generated portion (not the input).
    """
    model, tokenizer = await _load_model_and_tokenizer()
    loop = asyncio.get_running_loop()

    def _generate():
        # Use tokenizer.pad() for correct left-padding (handles edge cases
        # and avoids the pad_token_id=0 fallback that can corrupt BOS tokens)
        padded = tokenizer.pad(
            {"input_ids": [t.input_ids for t in batch],
             "attention_mask": [t.attention_mask for t in batch]},
            padding=True,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,  # early termination for short answers
            )

        # Decode only the generated tokens (strip input prefix)
        generated = outputs[:, input_ids.shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return texts

    return await loop.run_in_executor(_gpu_pool, _generate)


def postprocess(item: PromptItem, response: str) -> GenerationResult:
    """Decode and run simple eval: check if expected substring is present."""
    match = None
    if item.expected:
        match = item.expected.lower() in response.lower()

    return GenerationResult(
        line_idx=item.line_idx,
        prompt=item.prompt,
        response=response.strip(),
        num_input_tokens=item.num_tokens,
        num_output_tokens=len(response.split()),
        expected=item.expected,
        match=match,
    )


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------


@alru_cache(maxsize=1)
async def get_pipeline() -> InferencePipeline[PromptItem, TokenizedPrompt, str, GenerationResult]:
    pipeline = InferencePipeline(
        preprocess_fn=preprocess,
        inference_fn=inference_batch,
        postprocess_fn=postprocess,
        target_batch_cost=TARGET_BATCH_TOKENS,
        max_batch_size=16,
        min_batch_size=4,           # avoid tiny batches that underutilize the GPU
        batch_timeout_s=0.2,
        max_queue_size=500,
        pipeline_depth=8,
    )
    await pipeline.start()
    return pipeline


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------


@worker.task(cache="auto", retries=2)
async def generate_responses(
    jsonl_file: flyte.io.File,
    chunk_id: str,
    max_new_tokens: int = 128,
) -> list[dict]:
    """Process a chunk of JSONL prompts through the LLM pipeline.

    Each line is expected to be a JSON object with at least a "prompt"
    field, and optionally an "expected" field for simple eval.
    """
    pipeline = await get_pipeline()

    # Download and parse JSONL
    local_path = await jsonl_file.download()
    items = []
    with open(local_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(PromptItem(
                prompt=data["prompt"],
                expected=data.get("expected", ""),
                line_idx=i,
            ))

    results = await pipeline.run_all(items)

    # Log stats
    matches = [r for r in results if r.match is True]
    total_with_expected = [r for r in results if r.match is not None]
    accuracy = len(matches) / len(total_with_expected) if total_with_expected else 0

    logger.info(
        "[%s] %d prompts | GPU util: %.1f%% | avg batch: %.1f tokens | accuracy: %.1f%% (%d/%d)",
        chunk_id,
        len(results),
        pipeline.stats.utilization * 100,
        pipeline.stats.avg_batch_cost,
        accuracy * 100,
        len(matches),
        len(total_with_expected),
    )

    return [
        {
            "line_idx": r.line_idx,
            "prompt": r.prompt[:200],  # truncate for readability
            "response": r.response,
            "num_output_tokens": r.num_output_tokens,
            "expected": r.expected,
            "match": r.match,
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Driver task
# ---------------------------------------------------------------------------


@driver.task(cache="auto")
async def batch_generate(
    jsonl_path: str = "prompts.jsonl",
    max_new_tokens: int = 128,
    chunk_size: int = 50,
) -> list[dict]:
    """Generate LLM responses for all prompts in a JSONL file.

    If no JSONL file is provided, creates a demo dataset with sample prompts.
    """
    # Create demo JSONL if needed
    if jsonl_path == "prompts.jsonl":
        jsonl_path = _create_demo_jsonl()

    # Read and chunk the JSONL
    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Loaded {len(lines)} prompts from {jsonl_path}")

    # Split into chunk files and upload
    tasks = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        chunk_id = f"chunk_{i // chunk_size:03d}"

        # Write chunk to temp file and upload
        chunk_path = os.path.join(tempfile.gettempdir(), f"{chunk_id}.jsonl")
        with open(chunk_path, "w") as f:
            f.write("\n".join(chunk_lines) + "\n")
        chunk_file = await flyte.io.File.from_local(chunk_path)

        with flyte.group(f"generate-{chunk_id}"):
            tasks.append(asyncio.create_task(
                generate_responses(chunk_file, chunk_id, max_new_tokens)
            ))

    all_results = await asyncio.gather(*tasks)
    flat = [r for chunk_results in all_results for r in chunk_results]

    # Summary
    matches = sum(1 for r in flat if r.get("match") is True)
    total_eval = sum(1 for r in flat if r.get("match") is not None)
    print(f"\nCompleted {len(flat)} generations")
    if total_eval:
        print(f"Eval accuracy: {matches}/{total_eval} ({matches/total_eval*100:.1f}%)")

    return flat


def _create_demo_jsonl() -> str:
    """Create a small demo JSONL with diverse prompts for testing."""
    prompts = [
        {"prompt": "What is the capital of France?", "expected": "Paris"},
        {"prompt": "What is 2 + 2?", "expected": "4"},
        {"prompt": "Translate 'hello' to Spanish.", "expected": "hola"},
        {"prompt": "What color is the sky on a clear day?", "expected": "blue"},
        {"prompt": "What is the largest planet in our solar system?", "expected": "Jupiter"},
        {"prompt": "What is the chemical symbol for water?", "expected": "H2O"},
        {"prompt": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
        {"prompt": "What is the speed of light in km/s approximately?", "expected": "300"},
        {"prompt": "Name the first element on the periodic table.", "expected": "Hydrogen"},
        {"prompt": "What year did World War II end?", "expected": "1945"},
        {"prompt": "What is the square root of 144?", "expected": "12"},
        {"prompt": "What continent is Brazil in?", "expected": "South America"},
        {"prompt": "What is the boiling point of water in Celsius?", "expected": "100"},
        {"prompt": "Who painted the Mona Lisa?", "expected": "Vinci"},
        {"prompt": "What is the longest river in the world?", "expected": "Nile"},
        {"prompt": "Summarize the theory of relativity in one sentence."},
        {"prompt": "Explain how a neural network learns."},
        {"prompt": "Write a haiku about programming."},
        {"prompt": "What are the benefits of async programming in Python?"},
        {"prompt": "Describe the difference between TCP and UDP."},
    ]
    # Repeat to get enough volume for meaningful batching
    all_prompts = prompts * 5  # 100 prompts

    path = os.path.join(tempfile.gettempdir(), "demo_prompts.jsonl")
    with open(path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(batch_generate, max_new_tokens=128)
    print(run.url)
