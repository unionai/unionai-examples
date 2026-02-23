# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "colpali-engine>=0.3.1",
#     "transformers>=4.41",
#     "sentencepiece>=0.2",
#     "torch>=2.0",
#     "pillow>=10",
#     "datasets>=2.18",
#     "rank-bm25>=0.2",
#     "numpy>=1.26",
#     "pytesseract>=0.3",
#     "pydantic>=2.0",
#     "flyte>=2.0.0",
# ]
# ///
"""
Multimodal Retrieval Evaluation Pipeline

This tutorial is an experiment framework for benchmarking visual document
retrieval approaches on the ViDoRe benchmark. Each experiment is defined by
an ExperimentConfig; the pipeline fans them out as concurrent Flyte tasks and
returns a ranked comparison table with an interactive HTML report.

The corpus is a set of PDF page images; queries are plain-text questions. Each
retrieval method must find the page that answers each question — no text is
provided to the model, only the raw image.

  ColPali-v1.2  — patch-level multi-vector embeddings from a VLM (PaliGemma).
                  No OCR. The model produces one vector per image patch
                  (~1024 per page). MaxSim late-interaction scoring finds the
                  best matching patch for each query token.

  SigLIP-SO400M — single global embedding per page from Google's 2023 CLIP
                  successor. One matrix multiply per query; fast and effective
                  but a single vector cannot localise fine-grained regions.

  OCR + BM25    — text-only baseline. Tesseract extracts text, BM25 matches
                  keywords. Strong on text-dense pages; fails on charts,
                  tables, and figures where content is visual.

"""

import asyncio
import enum
import json
import math
import os
import tempfile
from functools import lru_cache
from io import BytesIO
from itertools import islice

import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

import flyte
import flyte.report
from flyte.io import File

# ─────────────────────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────────────────────

# One Docker image for all tasks. The PEP 723 header defines Python deps;
# with_apt_packages adds the Tesseract binary needed for the OCR baseline.
image = (
    flyte.Image.from_uv_script(__file__, name="vidore-eval")
    .with_apt_packages("tesseract-ocr", "ca-certificates")
    # unionai-reuse installs the unionai-actor-bridge binary required by ReusePolicy.
    # Without it every reusable container exits with StartError (exit code 128).
    .with_pip_packages("unionai-reuse>=0.1.9")
)

# GPU environment for ColPali and SigLIP image encoding.
#
# ReusePolicy keeps up to 3 warm GPU containers alive between task calls.
# Without it, every task invocation cold-starts a new container and downloads
# ColPali-v1.2 (~7 GB) from scratch. With it, the container — and the model
# weights already loaded into VRAM — is reused for the next task dispatch.
#
#   replicas=3      one warm container per concurrent GPU experiment
#   concurrency=1   one task at a time per container (GPU is not shared)
#   idle_ttl=120    keep alive 2 min after the last task finishes
#   scaledown_ttl=60 scale to zero after 1 min of complete inactivity
gpu_indexer = flyte.TaskEnvironment(
    name="vidore-gpu-indexer",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
    reusable=flyte.ReusePolicy(
        replicas=3,
        concurrency=1,
        idle_ttl=120,
        scaledown_ttl=60,
    ),
)

# Driver: orchestration, OCR baseline, BM25 search, evaluation, and reporting.
# depends_on ensures the shared Docker image is built before both environments
# try to schedule tasks.
driver = flyte.TaskEnvironment(
    name="vidore-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_indexer],
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration types
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalModel(str, enum.Enum):
    """Retrieval backend to evaluate."""

    COLPALI = "colpali-v1.2"  # multi-vector patch embeddings, MaxSim
    SIGLIP = "siglip-so400m"  # single-vector global embedding, cosine sim
    OCR_BM25 = "ocr+bm25"  # text extracted by Tesseract, ranked by BM25


class ExperimentConfig(BaseModel):
    """
    All knobs for one retrieval experiment. Passed as a typed Flyte input.

    Because ExperimentConfig is a Pydantic model, Flyte serialises it
    alongside every task output — so you can always reconstruct which
    config produced which metric without maintaining a separate log.
    """

    name: str  # human-readable label shown in the comparison table
    model: RetrievalModel
    top_k: int = 5  # number of pages to retrieve per query


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


class PageQuery(BaseModel):
    """One retrieval query with its ground-truth page."""

    query_id: str
    text: str  # e.g. "What was revenue growth in Q3?"
    relevant_page_id: str  # one correct page per query


class PageDataset(BaseModel):
    """
    A corpus of document page images paired with text queries.

    page_ids:   unique page identifiers (derived from ViDoRe image filenames).
    page_files: the same pages stored in Flyte's blob store as JPEG File
                handles. Tasks read images directly from here; no live HTTP.
    queries:    text questions with ground-truth page IDs for evaluation.
    """

    page_ids: list[str]
    page_files: list[File]
    queries: list[PageQuery]

    class Config:
        arbitrary_types_allowed = True


class RetrievalResult(BaseModel):
    query_id: str
    ranked_page_ids: list[str]  # ordered best → worst


class Metrics(BaseModel):
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    k: int


class ExperimentResult(BaseModel):
    config: ExperimentConfig
    metrics: Metrics


class ComparisonReport(BaseModel):
    results: list[ExperimentResult]

    def best_by(self, metric: str = "recall_at_k") -> ExperimentResult:
        return max(self.results, key=lambda r: getattr(r.metrics, metric))

    def summary(self) -> str:
        header = f"{'Experiment':<30} {'Model':<18} {'Recall@K':>10} {'NDCG@K':>8} {'MRR':>7}"
        sep = "─" * len(header)
        rows = [header, sep]
        for r in sorted(self.results, key=lambda x: -x.metrics.recall_at_k):
            rows.append(
                f"{r.config.name:<30} "
                f"{r.config.model.value:<18} "
                f"{r.metrics.recall_at_k:>10.3f} "
                f"{r.metrics.ndcg_at_k:>8.3f} "
                f"{r.metrics.mrr:>7.3f}"
            )
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────────────────────
# These functions are at module level so they are shared across all tasks that
# run on the same warm container (via ReusePolicy). lru_cache(maxsize=1) means
# the model is loaded from disk/HuggingFace exactly once per container process
# and kept in GPU memory for every subsequent task dispatch to that container.


@lru_cache(maxsize=1)
def _colpali_model():
    """Load ColPali-v1.2 into GPU memory and cache the result."""
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    return model, processor, device


@lru_cache(maxsize=1)
def _siglip_model():
    """Load SigLIP SO400M into GPU memory and cache the result."""
    import torch
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-224").to(device)
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    return model, processor, device


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _batches(items: list, batch_size: int):
    """Yield successive fixed-size batches from a list."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _load_npz(index_file: File) -> np.lib.npyio.NpzFile:
    """Download an index File to a local temp path and open with np.load."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        with index_file.open_sync("rb") as fh:
            tmp.write(fh.read())
        return np.load(tmp.name)


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — data loading
# ─────────────────────────────────────────────────────────────────────────────


@driver.task(cache="auto", retries=3)
async def load_vidore_pages(subset: str = "docvqa", max_pages: int = 200) -> PageDataset:
    """
    Stream a slice of a ViDoRe benchmark subset and store page images in Flyte.

    ViDoRe (Visual Document Retrieval Benchmark) contains PDF page images
    paired with text queries. Each sample = one (query, relevant page) pair.
    The retrieval task: given a query, find the correct page from the corpus.

    streaming=True reads only the rows requested via islice — no full-shard
    download, no rate limits. The first call uploads page images to Flyte's
    blob store and caches the PageDataset; every subsequent call with the same
    arguments returns the cached result instantly.

    retries=3 guards against transient HuggingFace network failures.

    Available subsets: "docvqa", "infovqa"
    Dataset: vidore/{subset}_test_subsampled
    """
    from datasets import load_dataset

    subset_map = {
        "docvqa": "vidore/docvqa_test_subsampled",
        "infovqa": "vidore/infovqa_test_subsampled",
    }
    dataset_name = subset_map.get(subset, f"vidore/{subset}_test_subsampled")
    ds = load_dataset(dataset_name, split="test", streaming=True)

    page_ids: list[str] = []
    page_files: list[File] = []
    queries: list[PageQuery] = []
    seen_pages: dict[str, str] = {}  # image_filename → page_id

    for i, row in enumerate(islice(ds, max_pages)):
        img = row.get("image")
        if not isinstance(img, PILImage.Image):
            continue
        filename: str = row.get("image_filename") or f"page_{i}"
        query_text: str = row.get("query", "")
        if not query_text:
            continue

        # Each unique page is uploaded to Flyte blob store exactly once.
        # Multiple queries can reference the same page (same image_filename).
        if filename not in seen_pages:
            page_id = f"{subset}_{len(page_ids):04d}"
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.convert("RGB").save(f.name, format="JPEG")
                page_file = await File.from_local(f.name)
            seen_pages[filename] = page_id
            page_ids.append(page_id)
            page_files.append(page_file)
        else:
            page_id = seen_pages[filename]

        queries.append(
            PageQuery(
                query_id=f"q{i:04d}",
                text=query_text,
                relevant_page_id=page_id,
            )
        )

    print(f"Loaded {len(page_ids)} unique pages, {len(queries)} queries", flush=True)
    return PageDataset(page_ids=page_ids, page_files=page_files, queries=queries)


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — indexing
# ─────────────────────────────────────────────────────────────────────────────


@gpu_indexer.task(cache="auto", retries=2)
async def index_colpali(page_ids: list[str], page_files: list[File]) -> File:
    """
    Encode every page with ColPali-v1.2 and save the multi-vector index.

    ColPali skips OCR entirely. It feeds the raw page image into PaliGemma
    (a vision-language model) and produces one embedding vector per image
    patch — roughly 1,024 patches per page, each of dimension 128.

    _colpali_model() is an lru_cache'd loader. On a cold container, it
    downloads and loads the model once. On a warm container (kept alive by
    ReusePolicy), it returns the already-loaded model instantly from cache —
    no repeated ~7 GB download.

    The index is stored as a .npz file in Flyte's blob store:
      embeddings — float32, shape (n_pages, n_patches, dim)
      page_ids   — matching page ID strings

    cache="auto" + retries=2: the result is stored permanently on success;
    transient failures (e.g. HuggingFace rate limits) are retried twice.
    """
    import torch

    model, processor, device = _colpali_model()

    all_embeddings: list[np.ndarray] = []
    n_batches = math.ceil(len(page_ids) / 4)

    for batch_idx, batch_files in enumerate(_batches(page_files, 4)):
        images: list[PILImage.Image] = []
        for f in batch_files:
            with f.open_sync("rb") as fh:
                images.append(PILImage.open(BytesIO(fh.read())).convert("RGB"))

        inputs = processor.process_images(images)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model(**inputs)  # (batch, n_patches, dim)

        all_embeddings.append(emb.cpu().float().numpy())
        print(f"ColPali: indexed batch {batch_idx + 1}/{n_batches}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (n_pages, n_patches, dim)
    out_path = os.path.join(tempfile.gettempdir(), "colpali_index.npz")
    np.savez(out_path, embeddings=embeddings, page_ids=np.array(page_ids))
    return await File.from_local(out_path)


@gpu_indexer.task(cache="auto", retries=2)
async def index_siglip(page_ids: list[str], page_files: list[File]) -> File:
    """
    Encode every page with SigLIP SO400M and save the single-vector index.

    SigLIP (2023) is Google's successor to CLIP, trained with sigmoid loss
    instead of softmax — avoiding the normalisation bottleneck that limits
    CLIP's scalability. Produces one global embedding per page.

    _siglip_model() caches the model across warm container reuses.

    The index is stored as a .npz file:
      embeddings — float32, shape (n_pages, dim), L2-normalised
      page_ids   — matching page ID strings
    """
    import torch

    model, processor, device = _siglip_model()

    all_embeddings: list[np.ndarray] = []
    n_batches = math.ceil(len(page_ids) / 8)

    for batch_idx, batch_files in enumerate(_batches(page_files, 8)):
        images: list[PILImage.Image] = []
        for f in batch_files:
            with f.open_sync("rb") as fh:
                images.append(PILImage.open(BytesIO(fh.read())).convert("RGB"))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            emb = outputs.pooler_output  # (batch, dim)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalise

        all_embeddings.append(emb.cpu().float().numpy())
        print(f"SigLIP: indexed batch {batch_idx + 1}/{n_batches}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (n_pages, dim)
    out_path = os.path.join(tempfile.gettempdir(), "siglip_index.npz")
    np.savez(out_path, embeddings=embeddings, page_ids=np.array(page_ids))
    return await File.from_local(out_path)


@driver.task(cache="auto")
async def extract_page_texts(page_files: list[File]) -> list[str]:
    """
    OCR every page with Tesseract to produce a text-only baseline.

    Returns one text string per page in the same order as page_files.
    Cached: the same corpus is OCR'd at most once across all experiments
    that use the OCR+BM25 backend.
    """
    import pytesseract

    texts: list[str] = []
    for i, f in enumerate(page_files):
        with f.open_sync("rb") as fh:
            img = PILImage.open(BytesIO(fh.read())).convert("RGB")
        text = pytesseract.image_to_string(img)
        texts.append(text)
        print(f"OCR: processed page {i + 1}/{len(page_files)}", flush=True)
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — search
# ─────────────────────────────────────────────────────────────────────────────


@gpu_indexer.task
async def search_colpali(
    index_file: File,
    queries: list[PageQuery],
    top_k: int,
) -> list[RetrievalResult]:
    """
    Retrieve pages using ColPali MaxSim late interaction.

    MaxSim score for page p given query q:
        score(q, p) = Σ_{t ∈ query tokens} max_{j ∈ page patches} (q_t · p_j)

    Batching: all queries are encoded in a single GPU forward pass instead of
    one pass per query. This is the primary GPU efficiency win — encoding is
    O(1) GPU calls regardless of query count. Scoring still loops per query to
    avoid the memory cost of materialising the full
    (n_queries x n_tokens x n_pages x n_patches) tensor at once.
    """
    import torch

    data = _load_npz(index_file)
    corpus_emb = torch.from_numpy(data["embeddings"])  # (n_pages, n_patches, dim)
    index_page_ids: list[str] = list(data["page_ids"])

    model, processor, device = _colpali_model()
    corpus_emb = corpus_emb.to(device, dtype=torch.float32)

    # Encode all queries in one batch — one GPU forward pass total.
    all_query_inputs = processor.process_queries([q.text for q in queries])
    all_query_inputs = {k: v.to(device) for k, v in all_query_inputs.items()}

    with torch.no_grad():
        all_query_embs = model(**all_query_inputs)  # (n_queries, n_tokens, dim)
        all_query_embs = all_query_embs.float()

    # Score pages per query. The einsum is kept per-query to avoid allocating a
    # (n_queries x n_tokens x n_pages x n_patches) intermediate tensor.
    results: list[RetrievalResult] = []
    for q, query_emb in zip(queries, all_query_embs):
        # "td,pjd->tpj": score each (token t, page p, patch j) triple.
        # max(dim=2) → best patch per (token, page).
        # sum(dim=0) → aggregate token votes into a single page score.
        scores = torch.einsum("td,pjd->tpj", query_emb, corpus_emb).max(dim=2).values.sum(dim=0)
        top_indices = scores.argsort(descending=True)[:top_k].cpu().tolist()
        results.append(
            RetrievalResult(
                query_id=q.query_id,
                ranked_page_ids=[index_page_ids[i] for i in top_indices],
            )
        )

    return results


@gpu_indexer.task
async def search_siglip(
    index_file: File,
    queries: list[PageQuery],
    top_k: int,
) -> list[RetrievalResult]:
    """
    Retrieve pages using SigLIP cosine similarity.

    Batching: all queries are encoded and scored in a single GPU call.
    SigLIP's single-vector embeddings make full vectorisation safe —
    the scores matrix (n_pages x n_queries) is small enough to materialise
    in one operation, unlike the multi-vector ColPali case.
    """
    import torch

    data = _load_npz(index_file)
    corpus_emb = torch.from_numpy(data["embeddings"])  # (n_pages, dim), L2-normalised
    index_page_ids: list[str] = list(data["page_ids"])

    model, processor, device = _siglip_model()
    corpus_emb = corpus_emb.to(device)

    # Encode all queries in one batch.
    text_inputs = processor(
        text=[q.text for q in queries],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        text_out = model.text_model(**text_inputs)
        query_embs = text_out.pooler_output  # (n_queries, dim)
        query_embs = query_embs / query_embs.norm(dim=-1, keepdim=True)

    # Score all (page, query) pairs in one matrix multiply.
    scores_matrix = corpus_emb @ query_embs.T  # (n_pages, n_queries)

    results: list[RetrievalResult] = []
    for i, q in enumerate(queries):
        top_indices = scores_matrix[:, i].argsort(descending=True)[:top_k].cpu().tolist()
        results.append(
            RetrievalResult(
                query_id=q.query_id,
                ranked_page_ids=[index_page_ids[j] for j in top_indices],
            )
        )

    return results


@driver.task
async def search_bm25(
    page_texts: list[str],
    page_ids: list[str],
    queries: list[PageQuery],
    top_k: int,
) -> list[RetrievalResult]:
    """
    Retrieve pages using BM25 over OCR'd text.

    The standard keyword-based baseline. No GPU required; strong on
    text-dense pages, weak on visual content that Tesseract cannot read.
    """
    tokenized = [text.lower().split() for text in page_texts]
    bm25 = BM25Okapi(tokenized)

    results: list[RetrievalResult] = []
    for q in queries:
        scores = bm25.get_scores(q.text.lower().split())
        ranked = sorted(range(len(page_ids)), key=lambda i: -scores[i])[:top_k]
        results.append(
            RetrievalResult(
                query_id=q.query_id,
                ranked_page_ids=[page_ids[i] for i in ranked],
            )
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — evaluation
# ─────────────────────────────────────────────────────────────────────────────


@driver.task
async def evaluate(
    results: list[RetrievalResult],
    ground_truth: list[PageQuery],
    k: int,
) -> Metrics:
    """
    Compute Recall@K, NDCG@K, and MRR for a single retrieval model.

    Recall@K  — was the correct page in the top-K results?
    NDCG@K    — normalised discounted cumulative gain; rewards earlier hits.
    MRR       — mean reciprocal rank of the first correct result.

    All three are averaged over all queries. Higher is better.
    """
    gt_map = {q.query_id: q.relevant_page_id for q in ground_truth}
    recall_vals, ndcg_vals, mrr_vals = [], [], []

    for r in results:
        relevant = gt_map.get(r.query_id, "")
        top = r.ranked_page_ids[:k]

        recall_vals.append(1.0 if relevant in top else 0.0)

        rels = [1 if pid == relevant else 0 for pid in top]
        idcg = _dcg([1])  # ideal: correct page at rank 1
        ndcg_vals.append(_dcg(rels) / idcg if idcg > 0 else 0.0)

        rr = 0.0
        for rank, pid in enumerate(r.ranked_page_ids, start=1):
            if pid == relevant:
                rr = 1.0 / rank
                break
        mrr_vals.append(rr)

    return Metrics(
        recall_at_k=float(np.mean(recall_vals)),
        ndcg_at_k=float(np.mean(ndcg_vals)),
        mrr=float(np.mean(mrr_vals)),
        k=k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tasks — report
# ─────────────────────────────────────────────────────────────────────────────


@driver.task(report=True)
async def generate_report(report: ComparisonReport) -> None:
    """
    Emit an interactive HTML report visible in the Flyte UI.

    report=True marks this task as a reporting task. Flyte renders the HTML
    returned via flyte.report.replace.aio() directly in the execution detail
    page — no separate dashboard or export step required.

    The report contains:
      - Summary cards: experiment count, best model, best Recall@K.
      - Grouped bar chart: Recall@K, NDCG@K, MRR side-by-side per experiment.
      - Ranked results table with all three metrics.
    """
    sorted_results = sorted(report.results, key=lambda r: -r.metrics.recall_at_k)
    best = sorted_results[0]

    labels = [r.config.name for r in sorted_results]
    recall_vals = [r.metrics.recall_at_k for r in sorted_results]
    ndcg_vals = [r.metrics.ndcg_at_k for r in sorted_results]
    mrr_vals = [r.metrics.mrr for r in sorted_results]

    table_rows = "".join(
        f"""
        <tr>
          <td>{r.config.name}</td>
          <td>{r.config.model.value}</td>
          <td>{r.metrics.recall_at_k:.3f}</td>
          <td>{r.metrics.ndcg_at_k:.3f}</td>
          <td>{r.metrics.mrr:.3f}</td>
          <td>{r.metrics.k}</td>
        </tr>"""
        for r in sorted_results
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Visual Document Retrieval — Results</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f0f2f5; color: #222; padding: 24px;
    }}
    h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
    .subtitle {{ color: #666; margin-bottom: 24px; font-size: 0.95em; }}
    .cards {{
      display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px;
    }}
    .card {{
      background: #fff; border-radius: 10px; padding: 18px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08); min-width: 160px;
    }}
    .card-value {{ font-size: 1.9em; font-weight: 700; color: #4f46e5; }}
    .card-label {{ font-size: 0.8em; color: #888; text-transform: uppercase;
                   letter-spacing: .04em; margin-top: 2px; }}
    .chart-box {{
      background: #fff; border-radius: 10px; padding: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 28px;
    }}
    .chart-box h2 {{ font-size: 1em; margin-bottom: 16px; color: #444; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    th {{
      background: #4f46e5; color: #fff; padding: 10px 14px;
      text-align: left; font-weight: 600;
    }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f8f8ff; }}
    tr:first-child td {{ font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Visual Document Retrieval — Experiment Comparison</h1>
  <p class="subtitle">ViDoRe benchmark &middot; {len(report.results)} experiment(s)</p>

  <div class="cards">
    <div class="card">
      <div class="card-value">{len(report.results)}</div>
      <div class="card-label">Experiments</div>
    </div>
    <div class="card">
      <div class="card-value">{best.config.name}</div>
      <div class="card-label">Best by Recall@K</div>
    </div>
    <div class="card">
      <div class="card-value">{best.metrics.recall_at_k:.3f}</div>
      <div class="card-label">Best Recall@{best.metrics.k}</div>
    </div>
    <div class="card">
      <div class="card-value">{best.metrics.ndcg_at_k:.3f}</div>
      <div class="card-label">Best NDCG@{best.metrics.k}</div>
    </div>
  </div>

  <div class="chart-box">
    <h2>Metrics by Experiment</h2>
    <canvas id="metricsChart" height="100"></canvas>
  </div>

  <div class="chart-box">
    <h2>Ranked Results</h2>
    <table>
      <thead>
        <tr>
          <th>Experiment</th><th>Model</th>
          <th>Recall@K</th><th>NDCG@K</th><th>MRR</th><th>K</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <script>
    new Chart(document.getElementById('metricsChart'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(labels)},
        datasets: [
          {{
            label: 'Recall@K',
            data: {json.dumps(recall_vals)},
            backgroundColor: 'rgba(79,70,229,0.75)',
            borderRadius: 4
          }},
          {{
            label: 'NDCG@K',
            data: {json.dumps(ndcg_vals)},
            backgroundColor: 'rgba(16,185,129,0.75)',
            borderRadius: 4
          }},
          {{
            label: 'MRR',
            data: {json.dumps(mrr_vals)},
            backgroundColor: 'rgba(245,158,11,0.75)',
            borderRadius: 4
          }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          y: {{ beginAtZero: true, max: 1.0,
               title: {{ display: true, text: 'Score' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""

    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment orchestration
# ─────────────────────────────────────────────────────────────────────────────


@driver.task
async def run_experiment(config: ExperimentConfig, dataset: PageDataset) -> ExperimentResult:
    """
    End-to-end retrieval pipeline for a single ExperimentConfig.

    Flyte v2's dynamic execution means this driver task can call GPU tasks
    (index_colpali, search_colpali) based on the runtime value of config.model
    — no static DAG wiring required. The if/elif is plain Python; Flyte
    schedules the selected sub-tasks on the appropriate environment.

    Caching: two experiments that share the same model and corpus (e.g. ColPali
    at top_k=5 and top_k=10) will hit the same cached index. GPU work is paid
    at most once per (model, corpus) pair across all experiments.

    flyte.group wraps each experiment in a named span in the Flyte UI, making
    it easy to compare latencies and drill into individual runs.
    """
    with flyte.group(config.name):
        if config.model == RetrievalModel.COLPALI:
            index_file = await index_colpali(dataset.page_ids, dataset.page_files)
            results = await search_colpali(index_file, dataset.queries, config.top_k)

        elif config.model == RetrievalModel.SIGLIP:
            index_file = await index_siglip(dataset.page_ids, dataset.page_files)
            results = await search_siglip(index_file, dataset.queries, config.top_k)

        else:  # RetrievalModel.OCR_BM25
            page_texts = await extract_page_texts(dataset.page_files)
            results = await search_bm25(page_texts, dataset.page_ids, dataset.queries, config.top_k)

        metrics = await evaluate(results, dataset.queries, config.top_k)

    return ExperimentResult(config=config, metrics=metrics)


@driver.task
async def compare_experiments(
    configs: list[ExperimentConfig],
    subset: str = "docvqa",
    max_pages: int = 200,
) -> ComparisonReport:
    """
    Fan out over all experiment configs and return a ranked comparison table.

    The dataset is loaded once and shared across all experiments. Each config
    runs as a concurrent Flyte task via asyncio.gather. Experiments that share
    a model reuse the cached index — you only pay GPU time for new work.

    On completion, generate_report emits an interactive Chart.js HTML report
    visible directly in the Flyte execution detail page.
    """
    dataset = await load_vidore_pages(subset=subset, max_pages=max_pages)

    # All experiments launch concurrently. Shared cached outputs (same model,
    # same corpus) are served from cache rather than recomputed.
    experiment_coros = [run_experiment(config=cfg, dataset=dataset) for cfg in configs]
    results: list[ExperimentResult] = list(await asyncio.gather(*experiment_coros))

    report = ComparisonReport(results=results)
    print(report.summary())
    best = report.best_by("recall_at_k")
    print(f"\nBest by Recall@{best.metrics.k}: {best.config.name}")

    # Emit the interactive HTML report in the Flyte UI.
    await generate_report(report)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flyte.init_from_config()

    # Define the experiment grid. Each ExperimentConfig is one point in the
    # design space. Adding a new model or varying top_k is one line here —
    # no task code changes required.
    #
    # ColPali appears twice with different top_k values. The cache ensures
    # index_colpali runs only once and both experiments share that result.
    configs = [
        ExperimentConfig(name="colpali-top5", model=RetrievalModel.COLPALI, top_k=5),
        ExperimentConfig(name="colpali-top10", model=RetrievalModel.COLPALI, top_k=10),
        ExperimentConfig(name="siglip-top5", model=RetrievalModel.SIGLIP, top_k=5),
        ExperimentConfig(name="ocr-bm25-top5", model=RetrievalModel.OCR_BM25, top_k=5),
    ]

    run = flyte.with_runcontext().run(
        compare_experiments,
        configs=configs,
        subset="docvqa",
        max_pages=200,
    )
    print(f"Run URL: {run.url}")
