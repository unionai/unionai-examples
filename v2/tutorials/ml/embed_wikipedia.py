# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte",
#    "sentence-transformers>=5.1.2",
#    "transformers>=4.41.0",
#    "huggingface-hub>=0.24",
#    "hf-transfer",
#    "datasets>=2.18",
# ]
# main = "embed_wikipedia"
# params = "batch_size=1024, shard='20231101.en'"
# ///

# {{docs-fragment imports}}
import asyncio
import concurrent.futures
import logging
import os
import tempfile
from collections import defaultdict
from typing import AsyncGenerator

import torch
from async_lru import alru_cache
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

import flyte
import flyte.io
from flyte.extras import DynamicBatcher

# {{/docs-fragment imports}}

logger = logging.getLogger(__name__)

# -- Thread pools ---------------------------------------------------------------
# A dedicated single-thread pool for GPU work prevents I/O threads from blocking
# the encode call.  A larger pool handles HuggingFace dataset downloads and
# iteration so that many shards can load data in parallel.
# {{docs-fragment thread-pools}}
_gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="io")
# {{/docs-fragment thread-pools}}

# -- Image & environments -------------------------------------------------------
# {{docs-fragment image}}
image = flyte.Image.from_uv_script(
    __file__, name="embed_wikipedia_image"
).with_pip_packages("unionai-reuse>=0.1.9")
# {{/docs-fragment image}}

# {{docs-fragment environments}}
worker = flyte.TaskEnvironment(
    name="embed_wikipedia_worker",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    # Each GPU replica accepts 8 concurrent tasks.  All 8 tasks share a single
    # model and DynamicBatcher, so texts from different shards are consolidated
    # into large GPU batches automatically.
    reusable=flyte.ReusePolicy(replicas=5, concurrency=8, idle_ttl=120, scaledown_ttl=120),
    secrets="HF_HUB_TOKEN",
)

driver = flyte.TaskEnvironment(
    name="embed_wikipedia_driver",
    image=image,
    resources=flyte.Resources(cpu=1, memory="4Gi", disk="16Gi"),
    secrets="HF_HUB_TOKEN",
    depends_on=[worker],
)
# {{/docs-fragment environments}}


# -- Model loading (shared across all concurrent tasks on a replica) -------------
# ``alru_cache`` ensures the model is loaded exactly once per process.  Because
# reusable containers run multiple tasks in the same process, every concurrent
# task reuses the same model without paying the load cost again.
# {{docs-fragment model}}
@alru_cache(maxsize=1)
async def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.half().to("cuda")
        # ``dynamic=True`` avoids recompilation when batch sizes vary at runtime.
        model[0].auto_model = torch.compile(model[0].auto_model, dynamic=True)
        # Warm up at a realistic batch size so the compiled kernels match production.
        model.encode(["warmup text " * 20] * 128, batch_size=128, convert_to_tensor=True)
    logger.warning(f"Model loaded on device: {model.device} (FP16: {torch.cuda.is_available()})")
    return model
# {{/docs-fragment model}}


# -- Dynamic batcher (shared across all concurrent tasks on a replica) -----------
# The DynamicBatcher collects text submissions from all 8 concurrent tasks and
# assembles them into large GPU batches.  Without it, each task would encode its
# own small batch independently, leaving the GPU idle between calls.
# See: https://www.union.ai/docs/v2/union/user-guide/run-scaling/batch-inference/
# {{docs-fragment batcher}}
@alru_cache(maxsize=1)
async def get_batcher(model_name: str = "all-MiniLM-L6-v2") -> DynamicBatcher[list[str], torch.Tensor]:
    model = await get_model(model_name)

    async def encode_batch(batches: list[list[str]]) -> list[torch.Tensor]:
        all_texts = [text for texts in batches for text in texts]
        loop = asyncio.get_running_loop()
        # Run the actual GPU inference on the dedicated GPU thread.
        all_embeddings = await loop.run_in_executor(
            _gpu_executor,
            lambda: model.encode(all_texts, convert_to_tensor=True, batch_size=1024).cpu(),
        )
        # Split results back to match the original submission groupings.
        results, offset = [], 0
        for texts in batches:
            results.append(all_embeddings[offset : offset + len(texts)])
            offset += len(texts)
        return results

    batcher = DynamicBatcher[list[str], torch.Tensor](
        process_fn=encode_batch,
        target_batch_cost=8_000,
        max_batch_size=64,
        batch_timeout_s=0.5,
        max_queue_size=1_000,
    )
    await batcher.start()
    return batcher
# {{/docs-fragment batcher}}


# -- Worker task: embed one parquet shard ----------------------------------------
# {{docs-fragment embed-shard}}
PIPELINE_DEPTH = 4


@worker.task(cache="auto", retries=4)
async def embed_shard(
    repo_id: str, filename: str, model_name: str, batch_size: int = 1024,
) -> flyte.io.File:
    batcher = await get_batcher(model_name)

    # Download the parquet file to local disk before iterating.
    # Streaming directly from HuggingFace adds 50-500 ms of HTTP latency per
    # chunk, which starves the GPU between batches.  Downloading first turns
    # iteration into a local disk read (microseconds per chunk).
    loop = asyncio.get_running_loop()
    local_path = await loop.run_in_executor(
        _io_executor,
        lambda: hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset",
            token=os.getenv("HF_HUB_TOKEN"),
            local_dir=os.path.join(tempfile.gettempdir(), "hf_cache"),
        ),
    )
    ds = load_dataset("parquet", data_files=local_path, split="train")

    # Producer/consumer pattern with a bounded queue.
    # The producer reads text batches and submits them to the batcher without
    # waiting for GPU results.  The consumer awaits results independently.
    # This keeps ``PIPELINE_DEPTH`` batches in-flight per coroutine, so the
    # batcher always has work queued when the GPU finishes a batch.
    all_embeddings: list[torch.Tensor] = []
    queue: asyncio.Queue = asyncio.Queue(maxsize=PIPELINE_DEPTH)

    async def producer():
        async for text_batch in _aiter_text_batches(ds, batch_size):
            future = await batcher.submit(text_batch, estimated_cost=len(text_batch))
            await queue.put(future)
        await queue.put(None)

    async def consumer():
        while True:
            future = await queue.get()
            if future is None:
                break
            all_embeddings.append(await future)
            if len(all_embeddings) % 10 == 0:
                print(f"  {filename}: {len(all_embeddings)} batches embedded")

    await asyncio.gather(producer(), consumer())

    shard_name = filename.replace("/", "_")
    out_path = os.path.join(tempfile.gettempdir(), f"{shard_name}.pt")
    if all_embeddings:
        torch.save(torch.cat(all_embeddings, dim=0), out_path)

    return await flyte.io.File.from_local(out_path)
# {{/docs-fragment embed-shard}}


# -- Async text batch iterator ---------------------------------------------------
# {{docs-fragment text-iter}}
async def _aiter_text_batches(ds, batch_size: int) -> AsyncGenerator[list[str], None]:
    """Yield batches of text from a HuggingFace dataset without blocking the event loop.

    ``ds.iter(batch_size=...)`` reads rows in bulk (one dict-of-lists per chunk),
    which is much faster than iterating row by row.
    """
    loop = asyncio.get_running_loop()
    sentinel = object()
    it = iter(ds.iter(batch_size=batch_size))
    while True:
        chunk = await loop.run_in_executor(_io_executor, lambda: next(it, sentinel))
        if chunk is sentinel:
            break
        texts = [t for t in chunk.get("text", []) if t]
        if texts:
            yield texts
# {{/docs-fragment text-iter}}


# -- Driver task: discover shards and fan out ------------------------------------
# {{docs-fragment driver}}
@driver.task(cache="auto")
async def embed_wikipedia(batch_size: int = 1024, shard: str = "all") -> list[flyte.io.File]:
    from huggingface_hub import HfApi

    repo_id = "wikimedia/wikipedia"
    model_name = "all-MiniLM-L6-v2"

    api = HfApi(token=os.getenv("HF_HUB_TOKEN"))
    info = api.dataset_info(repo_id)
    parquet_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet files in {repo_id}")

    # Group files by language shard (e.g. "20231101.en", "20231101.de").
    grouped = defaultdict(list)
    for filename in parquet_files:
        shard_id = filename.split("/")[0]
        grouped[shard_id].append(
            embed_shard(repo_id, filename, model_name=model_name, batch_size=batch_size),
        )

    if shard != "all":
        if shard not in grouped:
            raise ValueError(f"Shard {shard!r} not found.  Available: {sorted(grouped)}")
        grouped = {shard: grouped[shard]}

    tasks = []
    for shard_id, coros in grouped.items():
        print(f"  shard {shard_id}: {len(coros)} files")
        with flyte.group(f"shard-{shard_id}"):
            tasks.extend(asyncio.create_task(c) for c in coros)

    return await asyncio.gather(*tasks)
# {{/docs-fragment driver}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(embed_wikipedia, batch_size=1024, shard="20231101.en")
    print(run.url)
# {{/docs-fragment main}}
