# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "refine"
# params = "n_shards=2 docs_per_shard=2000 n_labels=300"
# ///
"""Refine raw web text into pretraining data: filter, deduplicate, decontaminate, and score it.

    N x refine_shard (warm CPU pool, cached per shard)
        Gopher / C4 / FineWeb heuristics, exact hash, MinHash band keys, 13-gram eval overlap
    global_dedup                exact and near duplicates across every shard, then decontamination
    sample_for_labeling         a few thousand surviving documents
    K x label_chunk (warm L40S) Qwen2.5-14B-Instruct under vLLM scores them on the FineWeb-Edu rubric
    train_classifier (L4)       frozen arctic-embed-m + a regression head, distilled from those labels
    N x score_shard  (warm L4)  the classifier scores every surviving document
    curate                      keep documents at or above the threshold; publish with a dataset card

This is how FineWeb-Edu was built: Llama-3-70B labeled 450,000 pages for educational value, and
a small classifier trained on those labels scored all 15 trillion tokens. Every stage before
curate is cached, so rerunning with a new threshold reuses the shards, the labels, and the scores.

    python refinery.py                     # 15 shards of 20,000 FineWeb documents, 4,000 LLM labels
    python refinery.py --smoke             # 2 shards of 2,000 documents, 300 labels
    python refinery.py --threshold 2.5     # same corpus, new threshold: everything but curate is cached
"""

import asyncio
import os
import zlib
import sys
import tempfile
import time
from dataclasses import dataclass, field

import classifier
import filters
import flyte
import flyte.artifacts as artifacts
import flyte.report
import labeler
import reports
from flyte.io import DataFrame, File

DATASET = "HuggingFaceFW/fineweb"
FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]  # about 1,000 documents per row group
CODE = ["filters.py", "labeler.py", "classifier.py", "reports.py", "viz.py"]

# {{docs-fragment images}}
# CPU stages: Arrow for the shards, xxhash for stable token hashes, SciPy for connected components.
cpu_image = flyte.Image.from_debian_base(name="refinery-cpu").with_pip_packages(
    "pyarrow==22.0.0", "pandas==2.3.3", "numpy==2.3.5", "scipy==1.16.3", "xxhash==3.6.0",
    "fsspec==2025.10.0", "aiohttp==3.13.2", "requests", "unionai-reuse==0.2.1",
)

# GPU stages: vLLM for the labeler. It brings torch and transformers, which the classifier uses.
gpu_image = flyte.Image.from_debian_base(name="refinery-gpu", python_version=(3, 12)).with_pip_packages(
    "vllm==0.33.0", "httpx>=0.27", "pandas", "pyarrow", "unionai-reuse==0.2.1",
)
# {{/docs-fragment images}}

# {{docs-fragment envs}}
# A warm pool of CPU workers takes the shard fan-out and the small global steps.
cpu_env = flyte.TaskEnvironment(
    name="refinery-cpu",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 8), concurrency=1, idle_ttl=600, scaledown_ttl=120),
    include=CODE,
)

# The labeler: one L40S that keeps the vLLM server and the model weights loaded between calls.
label_env = flyte.TaskEnvironment(
    name="refinery-labeler",
    image=gpu_image,
    resources=flyte.Resources(cpu=6, memory="40Gi", gpu="L40s:1", disk="60Gi"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=1, idle_ttl=600, scaledown_ttl=300),
    # vLLM's FlashInfer sampler compiles CUDA code at startup, which needs nvcc; the image has
    # only the CUDA runtime, so use the PyTorch sampler.
    env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
    include=CODE,
)

# The classifier is a 110M-parameter encoder, so a smaller GPU does. Two warm replicas split
# the scoring fan-out, and each keeps the encoder in memory between shards.
score_env = flyte.TaskEnvironment(
    name="refinery-scorer",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1", disk="40Gi"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=1, idle_ttl=600, scaledown_ttl=180),
    include=CODE,
)

driver_env = flyte.TaskEnvironment(
    name="refinery-driver",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[cpu_env, label_env, score_env],
    include=CODE,
)
# {{/docs-fragment envs}}


@dataclass
class ShardSpec:
    index: int
    file: str
    row_groups: list[int]


@dataclass
class EvalNgrams:
    hashes: File  # .npy of sorted uint64 13-gram hashes
    items: File  # parquet: benchmark, text, one row per eval question; ``hashes`` index into it
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ShardResult:
    index: int
    docs: File  # parquet of the documents that passed the heuristics, with their signatures
    raw: int
    kept: int
    rejected: dict[str, int] = field(default_factory=dict)
    examples: list[dict] = field(default_factory=list)
    contaminated: int = 0
    contam_examples: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class DedupResult:
    removed: File  # parquet: id, stage, kept_id
    after_exact: int
    after_near: int
    after_decontam: int
    examples: list[dict] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class LabelChunk:
    labels: File  # parquet: id, llm_score, justification, token counts
    n: int
    parsed: int
    seconds: float  # time spent labeling, without the model load
    startup_s: float  # vLLM start and model load, if this call paid for it
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ClassifierResult:
    head: File
    held_out: dict  # distilled classifier against held-out LLM labels
    public_vs_llm: dict  # public FineWeb-Edu classifier against the same held-out LLM labels
    ours_vs_public: dict
    points: list[list[float]] = field(default_factory=list)  # held out: [llm, ours, public]
    label_counts: list[int] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    n_train: int = 0
    curve: list[float] = field(default_factory=list)


@dataclass
class ScoredShard:
    scores: File  # parquet: id, edu_score
    n: int
    seconds: float  # time spent scoring, without the model load
    load_s: float


@dataclass
class Curated:
    kept: int
    scored: int
    tokens_kept: int
    tokens_scored: int
    histogram: list[int] = field(default_factory=list)  # 20 bins of 0.25 over [0, 5]
    examples: list[dict] = field(default_factory=list)


def _snippet(text: str, n: int = 600) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


# ── CPU stages ────────────────────────────────────────────────────────────────


# {{docs-fragment evals}}
EVAL_SETS = {
    "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    "arc-easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
}


@cpu_env.task(cache="auto", retries=2)
async def build_eval_ngrams(config: filters.FilterConfig) -> EvalNgrams:
    """Every 13-gram of every test question in GSM8K, MMLU and ARC, hashed like document text."""
    import numpy as np
    import pandas as pd

    rows = []
    for name, (repo, path) in EVAL_SETS.items():
        df = pd.read_parquet(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}")
        for q in df.to_dict("records"):
            question = q["question"]
            choices = q.get("choices")
            if isinstance(choices, dict):  # ARC: {"text": [...], "label": [...]}
                choices = choices["text"]
            extra = " ".join(map(str, choices)) if choices is not None else ""
            rows.append((name, question, extra))
    hashes, owners = [], []
    for i, (_, question, extra) in enumerate(rows):
        # The question alone, and the question followed by its answer options.
        grams = np.unique(np.concatenate([
            filters.ngram_hashes(filters.token_hashes(filters.tokens(t)), config.contam_ngram)
            for t in (question, f"{question} {extra}")
        ]))
        hashes.append(grams)
        owners.append(np.full(len(grams), i, dtype=np.int64))
    h, o = np.concatenate(hashes), np.concatenate(owners)
    order = np.argsort(h, kind="stable")
    d = tempfile.mkdtemp()
    np.save(f"{d}/ngrams.npy", np.stack([h[order], o[order].astype(np.uint64)]))
    pd.DataFrame(rows, columns=["benchmark", "text", "choices"]).to_parquet(f"{d}/items.parquet")
    counts = pd.Series([r[0] for r in rows]).value_counts().to_dict()
    print(f"{len(rows)} eval questions, {len(h)} distinct 13-grams")
    return EvalNgrams(await File.from_local(f"{d}/ngrams.npy"), await File.from_local(f"{d}/items.parquet"),
                      {k: int(v) for k, v in counts.items()})
# {{/docs-fragment evals}}


# {{docs-fragment shard}}
@cpu_env.task(cache="auto", retries=2)
async def refine_shard(spec: ShardSpec, config: filters.FilterConfig, evals: EvalNgrams) -> ShardResult:
    """Read one shard of raw web text, drop documents that fail a heuristic, and compute what the
    global steps need: an exact-match hash, MinHash band keys, and any overlap with an eval set."""
    import re

    import fsspec
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    started = time.monotonic()
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{spec.file}"
    with fsspec.open(url, "rb", block_size=16 * 2**20) as f:
        table = pq.ParquetFile(f).read_row_groups(spec.row_groups, columns=["id", "url", "dump", "token_count", "text"])
    eval_grams = np.load(await evals.hashes.download())
    eval_items = pd.read_parquet(await evals.items.download())

    rows, rejected, examples, contam_examples = [], {}, [], []
    for doc in table.to_pylist():
        text = doc["text"]
        words = filters.tokens(text)
        hashes = filters.token_hashes(words)
        reason = filters.heuristic_reject(text, config, words, hashes)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            if rejected[reason] <= 2:
                examples.append({"stage": "heuristics", "rule": reason, "url": doc["url"], "text": _snippet(text)})
            continue
        keys = filters.band_keys(filters.minhash(hashes, config), config)
        # Decontamination: does any 13-gram of the document appear in an eval question?
        grams = filters.ngram_hashes(hashes, config.contam_ngram)
        pos = np.searchsorted(eval_grams[0], grams).clip(max=eval_grams.shape[1] - 1)
        hits = np.flatnonzero(eval_grams[0][pos] == grams)
        contam = ""
        if len(hits):
            item = eval_items.iloc[int(eval_grams[1][pos[hits[0]]])]
            contam = item["benchmark"]
            spans = [m.span() for m in re.finditer(r"\w+", text)]
            first = min(hits[0], len(spans) - config.contam_ngram)
            start, end = spans[first][0], spans[first + config.contam_ngram - 1][1]
            contam_examples.append({
                "url": doc["url"], "benchmark": contam, "question": _snippet(item["text"], 400),
                "match": text[start:end], "matched_ngrams": int(len(np.unique(grams[hits]))),
                "context": _snippet(text[max(0, start - 250):end + 250], 700),
            })
        rows.append({
            "id": doc["id"], "url": doc["url"], "dump": doc["dump"], "token_count": doc["token_count"], "text": text,
            "exact": filters.exact_hash(text), "contam": contam,
            **{f"b{i}": k for i, k in enumerate(keys)},
        })
    out = os.path.join(tempfile.mkdtemp(), f"shard-{spec.index:03d}.parquet")
    pd.DataFrame(rows).to_parquet(out, index=False)
    seconds = time.monotonic() - started
    print(f"shard {spec.index}: {table.num_rows} read, {len(rows)} kept, {len(contam_examples)} contaminated, {seconds:.0f}s")
    return ShardResult(spec.index, await File.from_local(out), table.num_rows, len(rows), rejected, examples,
                       len(contam_examples), contam_examples[:20], round(seconds, 1))
# {{/docs-fragment shard}}


# {{docs-fragment dedup}}
@cpu_env.task(cache="auto")
async def global_dedup(shards: list[ShardResult], config: filters.FilterConfig) -> DedupResult:
    """Remove exact duplicates, then near duplicates, across every shard; then contaminated documents.

    Two documents are near-duplicate candidates when they share any MinHash band key. Candidates
    are joined into clusters (connected components), and each cluster keeps its first document."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    started = time.monotonic()
    bands = [f"b{i}" for i in range(config.bands)]
    frames = []
    for s in shards:
        t = pq.read_table(await s.docs.download(), columns=["id", "url", "exact", "contam", *bands]).to_pandas()
        t["shard"] = s.index
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    n = len(df)

    exact_dup = df.duplicated("exact", keep="first").to_numpy()
    first_of_hash = df.groupby("exact")["id"].transform("first").to_numpy()
    alive = np.flatnonzero(~exact_dup)

    src, dst = [], []
    for b in bands:
        keys = df[b].to_numpy()[alive]
        order = np.argsort(keys, kind="stable")
        same = keys[order][1:] == keys[order][:-1]
        src.append(alive[order[:-1][same]])
        dst.append(alive[order[1:][same]])
    src, dst = np.concatenate(src), np.concatenate(dst)
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    _, cluster = connected_components(graph, directed=False)
    rep = pd.Series(np.arange(n)[alive], index=alive).groupby(cluster[alive]).transform("min")
    near_dup = np.zeros(n, dtype=bool)
    near_dup[alive] = rep.to_numpy() != alive
    rep_of = np.arange(n)
    rep_of[alive] = rep.to_numpy()

    after_near = ~exact_dup & ~near_dup
    contaminated = after_near & (df["contam"].to_numpy() != "")
    stage = np.where(exact_dup, "exact duplicate", np.where(near_dup, "near duplicate",
                     np.where(contaminated, "contaminated", "")))
    removed = pd.DataFrame({"id": df["id"], "stage": stage,
                            "kept_id": np.where(exact_dup, first_of_hash, df["id"].to_numpy()[rep_of])})
    removed = removed[removed["stage"] != ""]
    out = os.path.join(tempfile.mkdtemp(), "removed.parquet")
    removed.to_parquet(out, index=False)

    # Examples for the report: a few duplicate pairs, with the text of both sides.
    pairs = pd.concat([removed[removed.stage == "exact duplicate"].head(3), removed[removed.stage == "near duplicate"].head(6)])
    wanted = set(pairs["id"]) | set(pairs["kept_id"])
    text = {}
    for s in shards:
        t = pq.read_table(await s.docs.download(), columns=["id", "url", "text"]).to_pandas()
        for r in t[t["id"].isin(wanted)].itertuples():
            text[r.id] = (r.url, r.text)
    examples = [{"stage": p.stage, "url": text[p.id][0], "text": _snippet(text[p.id][1], 400),
                 "kept_url": text[p.kept_id][0], "kept_text": _snippet(text[p.kept_id][1], 400)}
                for p in pairs.itertuples() if p.id in text and p.kept_id in text]
    sizes = pd.Series(cluster[alive]).value_counts()
    clusters = [{"size": int(c), "url": df["url"].iloc[int(alive[np.flatnonzero(cluster[alive] == k)[0]])]}
                for k, c in sizes[sizes > 1].head(5).items()]
    result = DedupResult(await File.from_local(out), int(n - exact_dup.sum()), int(after_near.sum()),
                         int((after_near & ~contaminated).sum()), examples, clusters, round(time.monotonic() - started, 1))
    print(f"{n} in, {result.after_exact} after exact, {result.after_near} after near, {result.after_decontam} after decontamination")
    return result
# {{/docs-fragment dedup}}


async def _survivors(docs: File, removed: File, columns: list[str]):
    import pandas as pd

    gone = set(pd.read_parquet(await removed.download(), columns=["id"])["id"])
    df = pd.read_parquet(await docs.download(), columns=columns)
    return df[~df["id"].isin(gone)]


@cpu_env.task(cache="auto")
async def sample_for_labeling(shards: list[ShardResult], dedup: DedupResult, n: int, chunks: int, seed: int = 0) -> list[File]:
    """A uniform sample of the surviving documents, split into chunks for the labeler."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    total = dedup.after_decontam
    take = set(rng.choice(total, size=min(n, total), replace=False).tolist())
    picked, offset = [], 0
    for s in shards:
        df = await _survivors(s.docs, dedup.removed, ["id", "text"])
        mine = [i - offset for i in range(offset, offset + len(df)) if i in take]
        picked.append(df.iloc[mine])
        offset += len(df)
    sample = pd.concat(picked).sample(frac=1.0, random_state=seed)
    d, files = tempfile.mkdtemp(), []
    for k, part in enumerate(np.array_split(np.arange(len(sample)), chunks)):
        path = f"{d}/chunk-{k}.parquet"
        sample.iloc[part].to_parquet(path, index=False)
        files.append(await File.from_local(path))
    return files


# ── GPU stages ────────────────────────────────────────────────────────────────


# {{docs-fragment label-task}}
@label_env.task(cache=flyte.Cache("auto", salt=labeler.VERSION), retries=1)
async def label_chunk(chunk: File, model: str) -> LabelChunk:
    """Score a chunk of documents on the FineWeb-Edu rubric with the LLM. The first call on a
    replica starts vLLM; the rest find it warm."""
    import pandas as pd

    server, started_here = labeler.get_server(model)
    df = pd.read_parquet(await chunk.download())
    t0 = time.monotonic()
    answers = await labeler.label_all(df["text"].tolist(), model)
    seconds = time.monotonic() - t0
    out = pd.DataFrame(answers)
    out.insert(0, "id", df["id"].to_numpy())
    path = os.path.join(tempfile.mkdtemp(), "labels.parquet")
    out.to_parquet(path, index=False)
    print(f"{len(df)} documents in {seconds:.0f}s, {out.llm_score.notna().sum()} with a score")
    return LabelChunk(await File.from_local(path), len(df), int(out.llm_score.notna().sum()), round(seconds, 2),
                      round(server.startup_s, 1) if started_here else 0.0,
                      int(out.prompt_tokens.sum()), int(out.completion_tokens.sum()))
# {{/docs-fragment label-task}}


# {{docs-fragment train-task}}
@score_env.task(cache=flyte.Cache("auto", salt=classifier.VERSION), retries=1)
async def train_classifier(chunks: list[File], labels: list[LabelChunk], held_out_share: float = 0.2) -> ClassifierResult:
    """Distill the LLM's labels into the encoder + head classifier, and measure it on held-out
    labels next to the public FineWeb-Edu classifier."""
    import pandas as pd
    import torch

    docs = pd.concat([pd.read_parquet(await c.download()) for c in chunks])
    lab = pd.concat([pd.read_parquet(await l.labels.download()) for l in labels])
    df = docs.merge(lab, on="id").dropna(subset=["llm_score"]).reset_index(drop=True)
    df["llm_score"] = df["llm_score"].astype(float)
    # A stable split by document id, so a rerun holds out the same documents.
    test = df["id"].map(lambda s: zlib.crc32(s.encode()) % 1000 < held_out_share * 1000).to_numpy()

    x = classifier.embed(df["text"].tolist())
    head, curve = classifier.train_head(x[~test], df["llm_score"].to_numpy()[~test])
    held = df[test].reset_index(drop=True)
    held["ours"] = classifier.predict(head, x[test])
    held["public"] = classifier.public_scores(held["text"].tolist())
    llm, ours, public = held["llm_score"].to_numpy(), held["ours"].to_numpy(), held["public"].to_numpy()

    path = os.path.join(tempfile.mkdtemp(), "head.pt")
    torch.save(head.state_dict(), path)
    examples = [{"llm": int(r.llm_score), "ours": float(r.ours), "public": float(r.public),
                 "text": _snippet(r.text, 500), "justification": r.justification}
                for score in range(6) for r in held[held["llm_score"] == score].head(1).itertuples()]
    return ClassifierResult(
        head=await File.from_local(path),
        held_out=classifier.agreement(llm, ours),
        public_vs_llm=classifier.agreement(llm, public),
        ours_vs_public=classifier.agreement(public, ours),
        points=[[float(a), round(float(b), 3), round(float(c), 3)] for a, b, c in zip(llm, ours, public)],
        label_counts=[int((df["llm_score"] == s).sum()) for s in range(6)],
        examples=examples,
        n_train=int((~test).sum()),
        curve=[round(v, 4) for v in curve],
    )
# {{/docs-fragment train-task}}


# {{docs-fragment score-task}}
@score_env.task(cache=flyte.Cache("auto", salt=classifier.VERSION), retries=1)
async def score_shard(docs: File, removed: File, head: File) -> ScoredShard:
    """Score every surviving document in one shard with the distilled classifier."""
    import pandas as pd
    import torch

    df = await _survivors(docs, removed, ["id", "text"])
    model = classifier.make_head()
    model.load_state_dict(torch.load(await head.download(), weights_only=True))
    load_s = classifier.warm_up()  # zero when this replica already holds the encoder
    t0 = time.monotonic()
    scores = classifier.predict(model, classifier.embed(df["text"].tolist()))
    seconds = time.monotonic() - t0
    path = os.path.join(tempfile.mkdtemp(), "scores.parquet")
    pd.DataFrame({"id": df["id"].to_numpy(), "edu_score": scores.clip(0, 5)}).to_parquet(path, index=False)
    print(f"{len(df)} documents in {seconds:.0f}s")
    return ScoredShard(await File.from_local(path), len(df), round(seconds, 2), round(load_s, 1))
# {{/docs-fragment score-task}}


# {{docs-fragment curate}}
@cpu_env.task(cache="auto")
async def curate(shards: list[ShardResult], dedup: DedupResult, scores: list[ScoredShard], threshold: float) -> tuple[DataFrame, Curated]:
    """Join the scores to the surviving documents and keep those at or above the threshold."""
    import numpy as np
    import pandas as pd

    kept, hist, scored, tokens_scored = [], np.zeros(20, dtype=int), 0, 0
    low = []
    for s, sc in zip(shards, scores):
        df = await _survivors(s.docs, dedup.removed, ["id", "url", "dump", "token_count", "text"])
        df = df.merge(pd.read_parquet(await sc.scores.download()), on="id")
        hist += np.histogram(df["edu_score"], bins=20, range=(0, 5))[0]
        scored += len(df)
        tokens_scored += int(df["token_count"].sum())
        low.append(df[df["edu_score"] < threshold].sample(n=min(2, len(df)), random_state=s.index))
        kept.append(df[df["edu_score"] >= threshold])
    out = pd.concat(kept, ignore_index=True)
    out["edu_int_score"] = out["edu_score"].round().clip(0, 5).astype(int)
    examples = [{"stage": "classifier", "rule": f"score {r.edu_score:.2f}", "url": r.url, "text": _snippet(r.text)}
                for r in pd.concat(low).head(4).itertuples()]
    examples += [{"stage": "kept", "rule": f"score {r.edu_score:.2f}", "url": r.url, "text": _snippet(r.text)}
                 for r in out.nlargest(3, "edu_score").itertuples()]
    stats = Curated(len(out), scored, int(out["token_count"].sum()), tokens_scored, hist.tolist(), examples)
    return await DataFrame.from_local(out), stats
# {{/docs-fragment curate}}


# ── driver ───────────────────────────────────────────────────────────────────


def shard_specs(n_shards: int, docs_per_shard: int) -> list[ShardSpec]:
    """Spread shards over the sample's files, so they come from different Common Crawl dumps."""
    groups = max(docs_per_shard // 1000, 1)
    return [ShardSpec(i, FILES[i % len(FILES)], list(range((i // len(FILES)) * groups + 40, (i // len(FILES) + 1) * groups + 40)))
            for i in range(n_shards)]


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True)
async def refine(
    n_shards: int = 15,
    docs_per_shard: int = 20_000,
    n_labels: int = 4000,
    label_chunks: int = 4,
    threshold: float = 3.0,
    label_model: str = "Qwen/Qwen2.5-14B-Instruct",
) -> DataFrame:
    """Turn raw web text into a scored, filtered, deduplicated and decontaminated corpus."""
    config = filters.FilterConfig()
    timing: dict[str, float] = {}
    t = time.monotonic()

    with flyte.group("filter-shards"):
        evals = await build_eval_ngrams(config)
        shards = list(await asyncio.gather(*[refine_shard(s, config, evals) for s in shard_specs(n_shards, docs_per_shard)]))
    timing["filter shards"] = time.monotonic() - t
    await flyte.report.replace.aio(reports.live(config, shards, None, None, None, None, threshold, evals.counts), do_flush=True)

    t = time.monotonic()
    dedup = await global_dedup(shards, config)
    timing["dedup + decontaminate"] = time.monotonic() - t
    await flyte.report.replace.aio(reports.live(config, shards, dedup, None, None, None, threshold, evals.counts), do_flush=True)

    t = time.monotonic()
    with flyte.group("llm-labels"):
        chunks = await sample_for_labeling(shards, dedup, n_labels, label_chunks)
        labels = []
        for c in chunks:  # one warm replica: the first call loads the model, the rest reuse it
            labels.append(await label_chunk(c, label_model))
    timing["LLM labels"] = time.monotonic() - t

    t = time.monotonic()
    clf = await train_classifier(chunks, labels)
    timing["train classifier"] = time.monotonic() - t
    await flyte.report.replace.aio(reports.live(config, shards, dedup, labels, clf, None, threshold, evals.counts), do_flush=True)

    t = time.monotonic()
    with flyte.group("score-shards"):
        scores = list(await asyncio.gather(*[score_shard(s.docs, dedup.removed, clf.head) for s in shards]))
    timing["score every document"] = time.monotonic() - t

    t = time.monotonic()
    dataset, curated = await curate(shards, dedup, scores, threshold)
    timing["curate"] = time.monotonic() - t

    summary = reports.summarize(config, shards, dedup, labels, clf, scores, curated, threshold, timing, label_model)
    await flyte.report.replace.aio(reports.final(summary, shards, dedup, clf, curated, evals.counts), do_flush=True)
    flyte.report.get_tab("Rejected documents").replace(reports.rejected_tab(shards, dedup, curated))
    flyte.report.get_tab("LLM labels and the classifier").replace(reports.labels_tab(clf, labels))
    await flyte.report.flush.aio()

    card = await artifacts.Card.create_from.aio(content=reports.card(summary), format="html", card_type="data")
    metadata = artifacts.Metadata(
        name="edu-web-corpus",
        description=f"{curated.kept:,} FineWeb documents with an educational score of at least {threshold}, "
                    f"from {summary['funnel'][0][1]:,} raw documents",
        kind="data",
        attrs={"threshold": str(threshold), "filters": config.version, "labeler": label_model,
               "classifier": f"{classifier.ENCODER} + head", "documents": str(curated.kept),
               "tokens": str(curated.tokens_kept), "source": f"{DATASET} sample-10BT"},
        card=card,
    )
    return artifacts.new(dataset, metadata)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    args = {}
    if "--threshold" in sys.argv:
        args["threshold"] = float(sys.argv[sys.argv.index("--threshold") + 1])
    if "--label-model" in sys.argv:
        args["label_model"] = sys.argv[sys.argv.index("--label-model") + 1]
    if "--smoke" in sys.argv:
        args.update(n_shards=2, docs_per_shard=2000, n_labels=300)
    run = flyte.run(refine, **args)
    print(run.url)
# {{/docs-fragment run}}
