# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b35",
#    "torch>=2.0",
#    "torchvision>=0.15",
#    "Pillow>=10.0",
#    "httpx",
#    "async-lru",
#    "datasets>=2.18",
# ]
# main = "batch_image_pipeline"
# params = "dataset_name='beans', split='test', max_images=200"
# ///

"""
Batch Image Classification Pipeline
====================================

Demonstrates a 3-stage async pipeline that maximizes GPU utilization by
overlapping I/O, CPU preprocessing, and GPU inference using
``InferencePipeline`` from ``flyte.extras``.

Architecture::

    [I/O: Download Images]            Runs on preprocess_executor (16 threads)
            |
    [CPU: Resize + Normalize]         Same executor — PIL/torchvision release the GIL
            |
    [GPU: model.forward()]            DynamicBatcher batches items, runs on gpu_pool (1 thread)
            |
    [Decode Labels + Confidence]      Event loop (lightweight)

Key patterns:
- ``InferencePipeline`` wires preprocess → DynamicBatcher → postprocess
- ``alru_cache`` singletons for model + pipeline (shared across concurrent tasks)
- ``ReusePolicy`` keeps warm containers with loaded models
- Multiple concurrent tasks on the same replica all feed one pipeline → bigger GPU batches

Usage::

    flyte run batch_image_pipeline.py classify_dataset
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO

import httpx
import torch
import torchvision.models as models
import torchvision.transforms as T
from async_lru import alru_cache
from PIL import Image

import flyte
import flyte.io
from flyte.extras import InferencePipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pools (module-level singletons, shared across concurrent tasks)
# ---------------------------------------------------------------------------

# I/O + CPU preprocessing share a pool — both release the GIL.
# A dedicated single-thread GPU pool prevents contention.
_io_cpu_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="io-cpu")
_gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

# ---------------------------------------------------------------------------
# Image & environments
# ---------------------------------------------------------------------------

image = flyte.Image.from_uv_script(
    __file__, name="batch_image_pipeline_image"
).with_pip_packages("unionai-reuse>=0.1.9")

worker = flyte.TaskEnvironment(
    name="image_pipeline_worker",
    image=image,
    resources=flyte.Resources(cpu=4, memory="8Gi", gpu="T4:1"),
    reusable=flyte.ReusePolicy(
        replicas=3,
        concurrency=4,  # 4 concurrent tasks per replica → 12 streams feeding 3 GPUs
        idle_ttl=120,
        scaledown_ttl=120,
    ),
)

driver = flyte.TaskEnvironment(
    name="image_pipeline_driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[worker],
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ImageItem:
    """A single image to classify."""
    url: str
    image_id: str


@dataclass
class ClassificationResult:
    """Final output after postprocessing."""
    image_id: str
    url: str
    top_label: str
    confidence: float
    top5: list[tuple[str, float]]


# ---------------------------------------------------------------------------
# Model loading (process-level singleton via alru_cache)
# ---------------------------------------------------------------------------

_preprocess_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@alru_cache(maxsize=1)
async def _load_model():
    """Load ResNet-50 once per process. Shared across all concurrent tasks."""
    loop = asyncio.get_running_loop()

    def _load():
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        if torch.cuda.is_available():
            model = model.half().cuda()
            # dynamic=False + reduce-overhead enables CUDA graphs for fixed shapes.
            # ResNet-50 input is always 224x224, only batch dim varies.
            model = torch.compile(model, dynamic=False, mode="reduce-overhead")
            # Warmup at all plausible batch sizes to avoid JIT spikes at runtime
            for bs in [1, 4, 8, 16, 32]:
                dummy = torch.randn(bs, 3, 224, 224, dtype=torch.float16, device="cuda")
                with torch.no_grad():
                    model(dummy)
        return model

    model = await loop.run_in_executor(_gpu_pool, _load)
    logger.warning("Model loaded on device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    return model


_IMAGENET_LABELS: list[str] = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]


# ---------------------------------------------------------------------------
# Pipeline stage functions
# ---------------------------------------------------------------------------

# Shared HTTP client for downloading images (created per-process)
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30, follow_redirects=True)
    return _http_client


async def preprocess(item: ImageItem) -> torch.Tensor:
    """Download an image and apply torchvision transforms.

    The download is async (httpx). The PIL resize/normalize runs on
    ``_io_cpu_pool`` to avoid blocking the event loop.
    """
    client = _get_http_client()
    resp = await client.get(item.url)
    resp.raise_for_status()

    loop = asyncio.get_running_loop()
    tensor = await loop.run_in_executor(
        _io_cpu_pool,
        lambda: _preprocess_transform(Image.open(BytesIO(resp.content)).convert("RGB")),
    )
    return tensor


@dataclass
class Top5Result:
    """Top-5 predictions computed on GPU, transferred as small tensors."""
    probs: torch.Tensor   # [5] float
    indices: torch.Tensor  # [5] int


async def inference_batch(batch: list[torch.Tensor]) -> list[Top5Result]:
    """Run model.forward() on a batch of preprocessed tensors.

    Stacks individual tensors, moves to GPU, runs inference, computes
    top-5 on-device (200x less D2H data than full logits), then
    transfers only the small result tensors back to CPU.
    """
    model = await _load_model()
    loop = asyncio.get_running_loop()

    def _forward():
        stacked = torch.stack(batch).half()
        if torch.cuda.is_available():
            # pin_memory + non_blocking enables async H2D transfer
            stacked = stacked.pin_memory().to("cuda", non_blocking=True)
        with torch.no_grad():
            logits = model(stacked)
        # Compute top-5 on GPU to minimize D2H transfer ([N,5] vs [N,1000])
        probs = torch.softmax(logits.float(), dim=1)
        top5_probs, top5_idx = torch.topk(probs, 5, dim=1)
        return [
            Top5Result(probs=top5_probs[i].cpu(), indices=top5_idx[i].cpu())
            for i in range(len(batch))
        ]

    return await loop.run_in_executor(_gpu_pool, _forward)


def postprocess(item: ImageItem, result: Top5Result) -> ClassificationResult:
    """Decode top-5 indices into human-readable labels."""
    top5 = [
        (_IMAGENET_LABELS[idx], prob)
        for idx, prob in zip(result.indices.tolist(), result.probs.tolist())
    ]
    return ClassificationResult(
        image_id=item.image_id,
        url=item.url,
        top_label=top5[0][0],
        confidence=top5[0][1],
        top5=top5,
    )


# ---------------------------------------------------------------------------
# Pipeline singleton (shared across concurrent tasks on a replica)
# ---------------------------------------------------------------------------


@alru_cache(maxsize=1)
async def get_pipeline() -> InferencePipeline[ImageItem, torch.Tensor, Top5Result, ClassificationResult]:
    pipeline = InferencePipeline(
        preprocess_fn=preprocess,
        inference_fn=inference_batch,
        postprocess_fn=postprocess,
        target_batch_cost=32,       # 1 cost per image (uniform size after resize)
        max_batch_size=32,
        min_batch_size=8,           # avoid pathologically small batches (T4 throughput
                                    # drops ~15x at batch=1 vs batch=32 for ResNet-50)
        batch_timeout_s=0.15,       # slightly longer to accumulate larger batches
        max_queue_size=1_000,
        pipeline_depth=16,          # up to 16 images preprocessed ahead of GPU
    )
    await pipeline.start()
    return pipeline


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------


@worker.task(cache="auto", retries=3)
async def classify_images(image_urls: list[str], chunk_id: str) -> list[dict]:
    """Classify a chunk of images through the 3-stage pipeline.

    Multiple concurrent calls on the same replica share one pipeline
    singleton, so the DynamicBatcher inside sees items from all streams.
    """
    pipeline = await get_pipeline()

    items = [
        ImageItem(url=url, image_id=f"{chunk_id}_{i}")
        for i, url in enumerate(image_urls)
    ]

    results = await pipeline.run_all(items)

    logger.info(
        "[%s] %d images classified | GPU utilization: %.1f%% | avg batch: %.1f",
        chunk_id,
        len(results),
        pipeline.stats.utilization * 100,
        pipeline.stats.avg_batch_size,
    )

    return [
        {
            "image_id": r.image_id,
            "url": r.url,
            "top_label": r.top_label,
            "confidence": r.confidence,
            "top5": [(label, round(conf, 4)) for label, conf in r.top5],
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Driver task
# ---------------------------------------------------------------------------


@driver.task(cache="auto")
async def classify_dataset(
    dataset_name: str = "beans",
    split: str = "test",
    max_images: int = 200,
    chunk_size: int = 50,
) -> list[dict]:
    """Load images from a HuggingFace dataset, fan out to GPU workers.

    Each chunk becomes a separate task call, routed to a warm replica.
    All concurrent tasks on the same replica share one InferencePipeline,
    keeping the GPU saturated.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if max_images:
        ds = ds.select(range(min(max_images, len(ds))))

    # Upload images and collect URLs
    import tempfile, os
    image_urls = []
    for i, row in enumerate(ds):
        img = row["image"]
        path = os.path.join(tempfile.gettempdir(), f"img_{i:05d}.jpg")
        img.convert("RGB").save(path)
        f = await flyte.io.File.from_local(path)
        image_urls.append(f.remote_path)

    print(f"Uploaded {len(image_urls)} images, chunking into groups of {chunk_size}")

    # Fan out to workers
    tasks = []
    for i in range(0, len(image_urls), chunk_size):
        chunk = image_urls[i : i + chunk_size]
        chunk_id = f"chunk_{i // chunk_size:03d}"
        with flyte.group(f"classify-{chunk_id}"):
            tasks.append(asyncio.create_task(
                classify_images(chunk, chunk_id)
            ))

    all_results = await asyncio.gather(*tasks)
    flat = [r for chunk_results in all_results for r in chunk_results]
    print(f"Classified {len(flat)} images total")
    return flat


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(classify_dataset, dataset_name="beans", split="test", max_images=200)
    print(run.url)
