# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "curate"
# params = ""
# ///
"""Curate long-form footage into short, filtered, captioned clips for training a video model.

    for each chunk of a few minutes of film, all chunks at once:
        cut_chunk     (CPU pool)   decode to a 360p proxy, detect shots, cut 2-10 s clips,
                                   sample frames, exposure and sharpness
        score_chunk   (GPU pool)   aesthetic (CLIP + LAION head), motion (RAFT), text (CRAFT)
        select                     thresholds -> keep or reject, first failing filter
        caption_clip  (VLM pool)   a small vision-language model captions each kept clip
    manifest (parquet) + sample clips -> dataset artifact with a card; a report with the funnel,
    contact sheets per filter, score histograms, captions and throughput per stage

This is the shape of the data pipelines behind open video generators such as Open-Sora 2.0
and Stable Video Diffusion, scaled down to five Blender open movies, 56 minutes of footage.
Every stage before captioning is cached per chunk, and each caption per clip, so changing a
threshold and running again recomputes only the selection and any newly kept captions.

    python curate.py                          # all five films: 31 chunks, 56 minutes
    python curate.py --smoke                  # 4 one-minute chunks from two films
    python curate.py --set min_aesthetic=5.2  # change a threshold; everything upstream is cached
"""

import asyncio
import json
import os
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields

import flyte
import flyte.artifacts as artifacts
import flyte.report
import reports
import scoring
import shots
from flyte.io import Dir, File
from sources import SOURCES, plan_chunks

VLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
# Cache keys hash each task's own source. Bump the salt when you change a helper module.
CACHE = flyte.Cache("auto", salt="v2")

# {{docs-fragment images}}
# unionai-reuse provides the bridge that reusable environments need.
cpu_image = (
    flyte.Image.from_debian_base(name="video-curation-cpu")
    .with_apt_packages("ffmpeg")
    .with_pip_packages(
        "scenedetect==0.6.7", "opencv-python-headless==4.11.0.86", "numpy<2.3", "pandas==2.3.3",
        "pyarrow==21.0.0", "Pillow==11.3.0", "unionai-reuse==0.2.1",
    )
)
# The default PyPI wheels of torch 2.9 carry CUDA 12.8.
gpu_image = (
    flyte.Image.from_debian_base(name="video-curation-gpu")
    .with_pip_packages("torch==2.9.1", "torchvision==0.24.1")
    .with_pip_packages(
        "transformers==4.57.1", "accelerate==1.10.1", "num2words==0.5.14", "easyocr==1.7.2",
        "opencv-python-headless==4.11.0.86", "numpy<2.3", "Pillow==11.3.0", "unionai-reuse==0.2.1",
    )
)
# {{/docs-fragment images}}

# {{docs-fragment envs}}
# Decoding and shot detection: a warm pool of CPU replicas that takes one chunk at a time each.
cpu_env = flyte.TaskEnvironment(
    name="video-curation-cut",
    image=cpu_image,
    resources=flyte.Resources(cpu=4, memory="8Gi", disk="20Gi"),
    reusable=flyte.ReusePolicy(replicas=(1, 5), concurrency=1, idle_ttl=600, scaledown_ttl=120),
)

# Scoring: CLIP, RAFT and the text detector load once per replica and stay on the GPU.
score_env = flyte.TaskEnvironment(
    name="video-curation-score",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=2, concurrency=1, idle_ttl=600),
)

# Captioning: a different model with a different lifetime, so its own pool. Several calls
# share a replica: while one clip generates, the next one's frames download.
caption_env = flyte.TaskEnvironment(
    name="video-curation-caption",
    image=gpu_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=(1, 2), concurrency=4, idle_ttl=600, scaledown_ttl=180),
)

driver_env = flyte.TaskEnvironment(
    name="video-curation-driver",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[cpu_env, score_env, caption_env],
)
# {{/docs-fragment envs}}


# {{docs-fragment cut}}
@cpu_env.task(cache=CACHE, retries=2)
async def cut_chunk(source: str, start_s: int, end_s: int) -> tuple[Dir, str]:
    """Decode one chunk of a film, cut it into clips, and sample each clip's frames."""
    src = SOURCES[source]
    out = tempfile.mkdtemp(prefix="chunk-")
    for sub in ("frames", "thumbs", "clips"):
        os.makedirs(f"{out}/{sub}")
    proxy = f"{tempfile.mkdtemp()}/proxy.mp4"
    began = t = time.time()
    seconds = {}

    shots.decode_window(src.url, start_s, end_s - start_s, proxy)
    seconds["decode"], t = time.time() - t, time.time()
    found = shots.detect_shots(proxy)
    clips, short = shots.shots_to_clips(found)
    seconds["shots"], t = time.time() - t, time.time()
    ids = [f"{source}-{start_s:04d}-{i:03d}" for i in range(len(clips))]
    stats = shots.sample_frames(proxy, clips, ids, out)
    shots.cut_clips(proxy, clips, ids, out)
    seconds["sample_and_cut"] = time.time() - t

    rows = [
        {"id": cid, "source": source, "chunk": f"{source}-{start_s:04d}", "shot": shot,
         "start_s": round(start_s + a / shots.FPS, 3), "end_s": round(start_s + b / shots.FPS, 3),
         "duration_s": round((b - a) / shots.FPS, 3), **s}
        for cid, (a, b, shot), s in zip(ids, clips, stats)
    ]
    shots.write_index(out, rows)
    summary = {"shots": len(found), "short_shots": short, "clips": len(rows), "footage_s": end_s - start_s,
               "seconds": {k: round(v, 1) for k, v in seconds.items()}, "replica": socket.gethostname(),
               "ran": [began, time.time()]}
    print(f"{source} {start_s}-{end_s}s: {len(found)} shots -> {len(rows)} clips in {sum(seconds.values()):.0f}s")
    return await Dir.from_local(out), json.dumps({"summary": summary, "clips": rows})
# {{/docs-fragment cut}}


# {{docs-fragment score}}
@score_env.task(cache=CACHE, retries=2)
async def score_chunk(chunk: Dir) -> str:
    """Aesthetic, motion and text scores for every clip of one chunk, on one GPU."""
    began = time.time()
    local = await chunk.download()
    with open(f"{local}/clips.json") as f:
        ids = [c["id"] for c in json.load(f)]
    cold = "clip" not in scoring._models
    t0 = time.time()
    rows, seconds = await asyncio.to_thread(scoring.score_clips, local, ids)
    print(f"scored {len(ids)} clips in {time.time() - t0:.1f}s on {scoring.replica()} {seconds}")
    return json.dumps({"scores": rows, "seconds": seconds, "replica": scoring.replica(),
                       "load_s": scoring._models["load_s"] if cold else 0.0, "ran": [began, time.time()]})
# {{/docs-fragment score}}


# {{docs-fragment caption}}
@caption_env.task(cache=CACHE, retries=2)
async def caption_clip(chunk: Dir, clip_id: str, model: str = VLM) -> str:
    """Caption one kept clip from its four sampled frames. Cached per clip."""
    began = time.time()
    local = tempfile.mkdtemp()
    os.makedirs(f"{local}/frames")
    await asyncio.gather(*[
        File.from_existing_remote(f"{chunk.path}/frames/{clip_id}_{k}a.jpg").download(f"{local}/frames/{clip_id}_{k}a.jpg")
        for k in range(len(shots.POSITIONS))
    ])
    text, seconds, load_s = await asyncio.to_thread(scoring.caption, model, local, clip_id)
    return json.dumps({"caption": text, "seconds": round(seconds, 3), "replica": scoring.replica(), "load_s": load_s,
                       "ran": [began, time.time()]})
# {{/docs-fragment caption}}


# {{docs-fragment select}}
@dataclass
class Thresholds:
    """Every clip must pass all five filters. Tune them on the report's histograms."""

    min_brightness: float = 20.0  # mean luma, 0-255: drops black frames and fades
    max_brightness: float = 235.0  # drops white flashes
    min_sharpness: float = 20.0  # variance of the Laplacian at 360p: drops blurry frames
    max_text: float = 0.02  # share of the frame covered by text: drops titles and credits
    min_aesthetic: float = 4.5  # LAION aesthetic score, 1-10
    min_motion: float = 0.1  # optical flow, px per frame at 256 px tall: drops still shots
    max_motion: float = 20.0  # drops whip pans and flashes that flow can't follow


FILTERS = ("exposure", "blur", "text", "aesthetic", "motion")


def select(clip: dict, t: Thresholds) -> dict:
    """Which filters a clip passes, the first one it fails, and whether it is kept."""
    passed = {
        "exposure": t.min_brightness <= clip["brightness"] <= t.max_brightness,
        "blur": clip["sharpness"] >= t.min_sharpness,
        "text": clip["text"] <= t.max_text,
        "aesthetic": clip["aesthetic"] >= t.min_aesthetic,
        "motion": t.min_motion <= clip["motion"] <= t.max_motion,
    }
    first_fail = next((f for f in FILTERS if not passed[f]), "")
    return {**{f"pass_{f}": ok for f, ok in passed.items()}, "first_fail": first_fail, "keep": not first_fail}
# {{/docs-fragment select}}


async def read_remote(path: str) -> bytes:
    async with File.from_existing_remote(path).open("rb") as f:
        return bytes(await f.read())


# {{docs-fragment driver}}
@driver_env.task(report=True, produces_artifacts=True)
async def curate(
    sources: list[str] = list(SOURCES),
    chunk_s: int = 120,
    head_s: int = 0,
    tail_s: int = 0,
    thresholds: Thresholds = Thresholds(),
    vlm: str = VLM,
    sample_size: int = 24,
    publish: bool = True,
) -> Dir:
    """Turn whole films (or their first ``head_s`` and last ``tail_s`` seconds) into a captioned clip dataset."""
    started = time.time()
    chunks = plan_chunks(sources, chunk_s, head_s, tail_s)
    clips: list[dict] = []
    summaries: dict[str, dict] = {}
    dirs: dict[str, Dir] = {}
    calls: list[dict] = []  # one row per task call: stage, timing, replica, model load, cache hit
    lock = asyncio.Lock()

    async def call(stage: str, label: str, coro) -> tuple:
        """Await one task call and record it for the timeline and the throughput table."""
        called = time.time()
        result = await coro
        info = json.loads(result[1] if stage == "cut" else result)
        info = info.get("summary", info)
        began, ended = info["ran"]
        # A cache hit returns the outputs of an earlier run, with that run's timestamps.
        row = {"stage": stage, "label": label, "replica": info["replica"], "load_s": info.get("load_s", 0.0),
               "cached": began < called - 1, "start": began, "end": ended}
        calls.append(row)
        return result, row

    async def process(chunk) -> None:
        with flyte.group(chunk.id):
            (d, cut), row = await call("cut", chunk.id, cut_chunk(chunk.source, chunk.start_s, chunk.end_s))
            cut = json.loads(cut)
            row.update(clips=len(cut["clips"]), busy_s=sum(cut["summary"]["seconds"].values()))
            scored = {"scores": []}
            if cut["clips"]:
                scored, row = await call("score", chunk.id, score_chunk(d))
                scored = json.loads(scored)
                row.update(clips=len(cut["clips"]), busy_s=sum(scored["seconds"].values()), per_scorer=scored["seconds"])
            mine = [{**c, **s} for c, s in zip(cut["clips"], scored["scores"])]
            for c in mine:
                c.update(select(c, thresholds))

            async def cap(c: dict) -> None:
                r, row = await call("caption", c["id"], caption_clip(d, c["id"], vlm))
                r = json.loads(r)
                row.update(clips=1, busy_s=r["seconds"])
                c["caption"] = r["caption"]

            await asyncio.gather(*[cap(c) for c in mine if c["keep"]])
        async with lock:
            dirs[chunk.id], summaries[chunk.id] = d, cut["summary"]
            clips.extend(mine)
            await flyte.report.replace.aio(
                reports.live(clips, summaries, calls, len(chunks), thresholds, started, time.time()), do_flush=True)

    await asyncio.gather(*[process(c) for c in chunks])
    wall_s = time.time() - started
    clips.sort(key=lambda c: c["id"])
    for c in clips:
        c.setdefault("caption", "")
        c["clip_uri"] = f"{dirs[c['chunk']].path}/clips/{c['id']}.mp4"

    # Thumbnails and frame strips for the report and the card, fetched only for the clips shown.
    shown = reports.pick_examples(clips, thresholds, sample_size)
    thumbs = dict(zip(shown["thumbs"], await asyncio.gather(*[
        read_remote(f"{dirs[c.rsplit('-', 1)[0]].path}/thumbs/{c}.jpg") for c in shown["thumbs"]])))
    frames = {cid: await asyncio.gather(*[read_remote(f"{dirs[cid.rsplit('-', 1)[0]].path}/frames/{cid}_{k}a.jpg")
                                          for k in range(len(shots.POSITIONS))]) for cid in shown["sample"]}

    out = build_dataset(clips, shown["sample"], frames, thresholds)
    for cid in shown["sample"]:
        with open(f"{out}/sample/{cid}.mp4", "wb") as f:
            f.write(await read_remote(next(c["clip_uri"] for c in clips if c["id"] == cid)))

    ctx = dict(clips=clips, summaries=summaries, calls=calls, thresholds=thresholds, wall_s=wall_s, origin=started,
               thumbs=thumbs, frames=frames, sample=shown["sample"], vlm=vlm)
    await flyte.report.replace.aio(reports.overview(**ctx))
    flyte.report.get_tab("Filters").replace(reports.filters_tab(**ctx))
    flyte.report.get_tab("Captions").replace(reports.captions_tab(**ctx))
    await flyte.report.flush.aio()

    dataset = await Dir.from_local(out)
    if not publish:
        return dataset
    kept = [c for c in clips if c["keep"]]
    card = await artifacts.Card.create_from.aio(content=reports.card(**ctx), format="html", card_type="data")
    metadata = artifacts.Metadata(
        name="blender-open-movie-clips",
        kind="data",
        description=f"{len(kept)} captioned 2-10 s clips ({sum(c['duration_s'] for c in kept) / 60:.1f} min) "
                    f"curated from {len(clips)} candidates in {len(sources)} Blender open movies (CC BY)",
        attrs={"clips": str(len(kept)), "candidates": str(len(clips)), "captioner": vlm,
               "thresholds": json.dumps(asdict(thresholds)), "license": "CC BY (see ATTRIBUTION.txt)"},
        card=card,
    )
    return artifacts.new(dataset, metadata)
# {{/docs-fragment driver}}


# {{docs-fragment manifest}}
def build_dataset(clips: list[dict], sample: list[str], frames: dict, thresholds: Thresholds) -> str:
    """The manifest as parquet, the sample clips with their frame strips, and the attribution."""
    import pandas as pd

    out = tempfile.mkdtemp(prefix="dataset-")
    os.makedirs(f"{out}/sample")
    columns = ["id", "source", "chunk", "shot", "start_s", "end_s", "duration_s", "brightness", "sharpness",
               "frame_diff", "text", "aesthetic", "motion", *[f"pass_{f}" for f in FILTERS], "first_fail",
               "keep", "caption", "clip_uri"]
    df = pd.DataFrame(clips)[columns]
    df["title"] = df["source"].map(lambda s: SOURCES[s].title)
    df["license"] = df["source"].map(lambda s: SOURCES[s].license)
    df.to_parquet(f"{out}/manifest.parquet", index=False)
    for cid in sample:
        with open(f"{out}/sample/{cid}.jpg", "wb") as f:
            f.write(reports.strip_jpeg(frames[cid]))
    with open(f"{out}/thresholds.json", "w") as f:
        json.dump(asdict(thresholds), f, indent=2)
    with open(f"{out}/ATTRIBUTION.txt", "w") as f:
        f.write("Clips cut from Blender Foundation open movies, used under Creative Commons Attribution licenses.\n\n")
        f.writelines(f"{s.title}: {s.credit}. {s.license}. Source: {s.url}\n" for s in SOURCES.values())
    return out
# {{/docs-fragment manifest}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    kwargs: dict = {}
    if "--smoke" in sys.argv:
        kwargs = dict(sources=["big_buck_bunny", "tears_of_steel"], head_s=60, tail_s=60, chunk_s=60, sample_size=8,
                      publish=False)
    if "--set" in sys.argv:  # e.g. --set min_aesthetic=5.2,max_text=0.01
        pairs = dict(p.split("=") for p in sys.argv[sys.argv.index("--set") + 1].split(","))
        names = {f.name for f in fields(Thresholds)}
        kwargs["thresholds"] = Thresholds(**{k: float(v) for k, v in pairs.items() if k in names})
    run = flyte.run(curate, **kwargs)
    print(run.url)
# {{/docs-fragment run}}
