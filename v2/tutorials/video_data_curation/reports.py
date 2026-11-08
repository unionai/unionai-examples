"""The curation report (overview, filters, captions), its live view, and the dataset card."""

import base64
import io
import json
import math
from dataclasses import asdict

import viz
from sources import SOURCES

FILTERS = ("exposure", "blur", "text", "aesthetic", "motion")
# What each filter looks at, how to label it, and the histogram's range.
METRIC = {
    "exposure": ("brightness", "mean brightness (0-255)", 0, 255),
    "blur": ("sharpness", "sharpness (variance of the Laplacian)", 0, 600),
    "text": ("text", "share of the frame covered by text", 0, 0.2),
    "aesthetic": ("aesthetic", "aesthetic score (LAION, 1-10)", 2, 8),
    "motion": ("motion", "optical flow (px per frame)", 0, 12),
}
STAGES = {"cut": "decode + shots (CPU)", "score": "score (GPU)", "caption": "caption (GPU)"}


def _bounds(f: str, t) -> list[tuple[float, str]]:
    return {
        "exposure": [(t.min_brightness, f"min {t.min_brightness:g}"), (t.max_brightness, f"max {t.max_brightness:g}")],
        "blur": [(t.min_sharpness, f"min {t.min_sharpness:g}")],
        "text": [(t.max_text, f"max {t.max_text:g}")],
        "aesthetic": [(t.min_aesthetic, f"min {t.min_aesthetic:g}")],
        "motion": [(t.min_motion, f"min {t.min_motion:g}"), (t.max_motion, f"max {t.max_motion:g}")],
    }[f]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def img(data: bytes, title: str = "", height: int = 72, fluid: bool = False) -> str:
    size = "width:100%;height:auto" if fluid else f"height:{height}px"
    return (f"<img src='data:image/jpeg;base64,{b64(data)}' title='{viz.esc(title)}' "
            f"style='{size};border-radius:4px;display:block'/>")


def strip_jpeg(frames: list[bytes], height: int = 120) -> bytes:
    """The four sampled frames of a clip side by side, as one JPEG."""
    from PIL import Image

    ims = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    ims = [i.resize((round(i.width * height / i.height), height)) for i in ims]
    strip = Image.new("RGB", (sum(i.width for i in ims) + 4 * (len(ims) - 1), height), (255, 255, 255))
    x = 0
    for i in ims:
        strip.paste(i, (x, 0))
        x += i.width + 4
    buf = io.BytesIO()
    strip.save(buf, "JPEG", quality=82)
    return buf.getvalue()


# ── selection views ──────────────────────────────────────────────────────────


def funnel(clips: list[dict]) -> list[tuple[str, int]]:
    """Clips left after each filter, applied in order."""
    left, out = len(clips), [("candidate clips", len(clips))]
    for f in FILTERS:
        left -= sum(c["first_fail"] == f for c in clips)
        out.append((f"after {f}", left))
    return out


def _spread(items: list, n: int) -> list:
    """``n`` items evenly spaced through a sorted list, so a contact sheet shows the whole range."""
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def examples(clips: list[dict], per_row: int = 6) -> dict[str, tuple[list[dict], list[dict]]]:
    """For each filter: clips it rejected first, and kept clips, both spread over the score range."""
    kept = [c for c in clips if c["keep"]]
    out = {}
    for f in FILTERS:
        key = METRIC[f][0]
        rejected = sorted((c for c in clips if c["first_fail"] == f), key=lambda c: c[key])
        out[f] = (_spread(rejected, per_row), _spread(sorted(kept, key=lambda c: c[key]), per_row))
    return out


def sample(clips: list[dict], n: int) -> list[str]:
    """Kept clips for the captions tab and the dataset sample: round robin over films, best aesthetic first."""
    by_source: dict[str, list[dict]] = {}
    for c in sorted((c for c in clips if c["keep"]), key=lambda c: -c["aesthetic"]):
        by_source.setdefault(c["source"], []).append(c)
    out = []
    while len(out) < n and any(by_source.values()):
        for lst in by_source.values():
            if lst and len(out) < n:
                out.append(lst.pop(0)["id"])
    return out


def pick_examples(clips: list[dict], thresholds, n: int) -> dict[str, list[str]]:
    ids = {c["id"] for rej, kep in examples(clips).values() for c in rej + kep}
    return {"thumbs": sorted(ids), "sample": sample(clips, n)}


# ── charts ───────────────────────────────────────────────────────────────────


def histogram(values: list[float], lo: float, hi: float, bounds: list[tuple[float, str]], passes, *,
              bins: int = 30, width: int = 560, height: int = 170, fmt=viz.compact, log: bool = False) -> str:
    """Counts per bin, with the threshold lines; bins whose center fails the filter are red.

    ``log`` uses a log scale for the counts, for distributions with one tall bin (most clips have no text).
    """
    if not values:
        return "<p class='note'>no data</p>"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(max(int((v - lo) / step), 0), bins - 1)] += 1  # out-of-range values land in the end bins
    top = max(viz.nice_ticks(0, max(counts))[-1], 1)
    if log:
        top = 10 ** math.ceil(math.log10(max(max(counts), 1) + 1))
    L, R, T, B = 40, 10, 22, 24
    W, H = width - L - R, height - T - B

    def X(v: float) -> float:
        return L + W * (min(max(v, lo), hi) - lo) / (hi - lo)

    def Y(v: float) -> float:
        frac = math.log10(1 + v) / math.log10(1 + top) if log else v / top
        return T + H * (1 - frac)

    yticks = [10 ** k for k in range(0, round(math.log10(top)) + 1)] if log else viz.nice_ticks(0, top)
    g = "".join(f"<line class='gl' x1='{L}' x2='{L + W}' y1='{Y(t):.1f}' y2='{Y(t):.1f}'/>"
                f"<text x='{L - 6}' y='{Y(t) + 4:.1f}' text-anchor='end'>{viz.compact(t)}</text>"
                for t in yticks)
    g += "".join(f"<text x='{X(t):.1f}' y='{T + H + 16}' text-anchor='middle'>{fmt(t)}</text>"
                 for t in viz.nice_ticks(lo, hi, 5) if lo <= t <= hi)
    bw = W / bins
    for i, n in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        ok = passes((a + b) / 2)
        c = viz.color(1) if ok else viz.color("critical")
        if n:
            g += (f"<rect x='{L + i * bw + 1:.1f}' y='{Y(n):.1f}' width='{max(bw - 2, 1):.1f}' height='{T + H - Y(n):.1f}' "
                  f"rx='2' style='fill:{c};opacity:{1 if ok else 0.75}'><title>{fmt(a)} to {fmt(b)}: {n} clips</title></rect>")
    for v, label in bounds:
        if lo <= v <= hi:
            g += (f"<line class='ref' x1='{X(v):.1f}' x2='{X(v):.1f}' y1='{T - 4}' y2='{T + H}'/>"
                  f"<text class='ink' x='{X(v):.1f}' y='{T - 8}' text-anchor='middle'>{viz.esc(label)}</text>")
    head = viz.legend([("kept by this filter", 1), ("rejected by this filter", "critical")])
    return f"{head}<svg viewBox='0 0 {width} {height}' style='max-width:{width}px' role='img'>{g}</svg>"


def _funnel_chart(clips: list[dict]) -> str:
    steps = funnel(clips)
    bars = [viz.Bar(label, n, 1 if i in (0, len(steps) - 1) else 7,
                    note=f"-{steps[i - 1][1] - n}" if i else "") for i, (label, n) in enumerate(steps)]
    return viz.bar_chart(bars, label_width=130, width=560)


def _by_film(clips: list[dict], summaries: dict) -> str:
    rows = []
    for name, s in SOURCES.items():
        mine = [c for c in clips if c["source"] == name]
        if not mine:
            continue
        sums = [v for k, v in summaries.items() if k.startswith(name + "-")]
        kept = sum(c["keep"] for c in mine)
        fails = {f: sum(c["first_fail"] == f for c in mine) for f in FILTERS}
        top = max(fails, key=fails.get)
        rows.append([viz.esc(s.title), f"{sum(x['footage_s'] for x in sums) / 60:.0f} min",
                     sum(x["shots"] for x in sums), len(mine), kept, f"{kept / len(mine):.0%}",
                     f"{top} ({fails[top]})" if fails[top] else "—"])
    return viz.table(["film", "footage", "shots", "clips", "kept", "kept %", "top rejection"], rows)


def throughput(calls: list[dict]) -> list[dict]:
    """Per stage: calls, cache hits, clips, busy time inside the tasks, replicas, model loads, wall-clock span."""
    out = []
    for stage, label in STAGES.items():
        mine = [c for c in calls if c["stage"] == stage and "busy_s" in c]
        if not mine:
            continue
        ran = [c for c in mine if not c["cached"]]
        busy = sum(c["busy_s"] for c in ran)
        clips = sum(c["clips"] for c in ran)
        loads = [c["load_s"] for c in ran if c.get("load_s")]
        out.append({"stage": label, "calls": len(mine), "cached": len(mine) - len(ran), "clips": clips,
                    "busy_min": busy / 60, "per_min": clips / (busy / 60) if busy else None,
                    "replicas": len({c["replica"] for c in ran}), "loads": len(loads),
                    "load_s": sum(loads) / max(len(loads), 1),
                    "wall_min": (max(c["end"] for c in ran) - min(c["start"] for c in ran)) / 60 if ran else 0.0})
    return out


def _throughput_table(calls: list[dict]) -> str:
    rows = [[r["stage"], r["calls"], r["cached"] or "—", r["clips"], f"{r['busy_min']:.1f}",
             f"<b>{r['per_min']:,.0f}</b>" if r["per_min"] else "—", r["replicas"] or "—",
             f"{r['loads']} ({r['load_s']:.0f}s each)" if r["loads"] else "—", f"{r['wall_min']:.1f}"]
            for r in throughput(calls)]
    per = {}
    for c in calls:
        for k, v in (c.get("per_scorer") or {}).items() if not c["cached"] else ():
            per.setdefault(k, [0, 0.0])
            per[k][0] += c["clips"]
            per[k][1] += v
    note = ("Clips and busy time count only calls that ran in this run, not cache hits. Busy time is measured "
            "inside each task, without scheduling, downloads or model loading. " + (
                "Per scorer, clips per GPU-minute: " + ", ".join(f"{k} {n / max(s / 60, 1e-9):,.0f}"
                                                                 for k, (n, s) in per.items()) + "." if per else ""))
    table = viz.table(["stage", "calls", "cache hits", "clips", "busy min", "clips / busy min", "replicas",
                       "model loads", "wall min"], rows)
    return table + f"<p class='note'>{note}</p>"


def _timeline(calls: list[dict], origin: float, now: float) -> str:
    """One lane per replica of each pool, from when the run started. Cache hits are left out."""
    spans = [viz.Span("run", "curate (driver)", origin, now, 8)]
    spans += [viz.Span(f"{STAGES[c['stage']]} · …{c['replica'][-5:]}", c["label"], c["start"], c["end"], i + 1)
              for i, st in enumerate(STAGES) for c in calls if c["stage"] == st and not c["cached"]]
    return viz.gantt(spans, packed=True, width=1000, label_width=230)


def _tiles(clips: list[dict], summaries: dict, wall_s: float, chunks_total: int | None = None) -> str:
    kept = [c for c in clips if c["keep"]]
    footage = sum(s["footage_s"] for s in summaries.values())
    first = ("Kept clips", f"{len(kept)}", f"of {len(clips)} candidates ({len(kept) / max(len(clips), 1):.0%})")
    items = [first,
             ("Kept footage", f"{sum(c['duration_s'] for c in kept) / 60:.1f} min", f"from {footage / 60:.0f} min of film"),
             ("Shots detected", f"{sum(s['shots'] for s in summaries.values())}",
              f"{sum(s['short_shots'] for s in summaries.values())} under 2 s dropped"),
             ("Wall clock", f"{wall_s / 60:.1f} min")]
    if chunks_total:
        items.insert(1, ("Chunks done", f"{len(summaries)} / {chunks_total}"))
    return viz.tiles(items, hero=True)


TITLE = "Curating Blender open movies into training clips"


def live(clips, summaries, calls, chunks_total, thresholds, origin, now) -> str:
    return viz.page(
        TITLE, "Each chunk is decoded and cut on the CPU pool, scored on the GPU pool, and its kept clips "
               "captioned on the captioning pool, all chunks at once.",
        _tiles(clips, summaries, now - origin, chunks_total),
        viz.row(viz.panel("Funnel so far", _funnel_chart(clips)), viz.panel("Per film", _by_film(clips, summaries))),
        viz.panel("Task calls over time", _timeline(calls, origin, now)),
    )


def overview(*, clips, summaries, calls, thresholds, wall_s, origin, **_) -> str:
    return viz.page(
        TITLE,
        f"{len(summaries)} chunks of {len({c['source'] for c in clips})} films, cut into shots, scored by five filters, and captioned. "
        "The Filters tab shows what each filter rejected, the Captions tab what the captioner wrote.",
        _tiles(clips, summaries, wall_s),
        viz.row(viz.panel("Funnel", _funnel_chart(clips), "Filters are applied in this order; "
                          "each clip is counted against the first filter it fails."),
                viz.panel("Per film", _by_film(clips, summaries))),
        viz.panel("Throughput per stage", _throughput_table(calls)),
        viz.panel("Task calls over time", _timeline(calls, origin, origin + wall_s),
                  "One lane per replica, from when the run started; the gap before the first call is replicas starting. "
                  "Scoring starts as soon as a chunk is cut, and captioning as soon as it is scored. Cache hits are not shown."),
        viz.panel("Thresholds", viz.code(json.dumps(asdict(thresholds), indent=2))),
    )


def _sheet(cs: list[dict], key: str, thumbs: dict, fmt) -> str:
    if not cs:
        return "<p class='note'>none</p>"
    cells = "".join(
        f"<div style='font-size:11px;color:var(--ink-2)'>{img(thumbs[c['id']], c['id'])}"
        f"<div><b>{fmt(c[key])}</b> · {viz.esc(SOURCES[c['source']].title)} {c['start_s']:.0f}s</div></div>"
        for c in cs if c["id"] in thumbs)
    return f"<div style='display:flex;flex-wrap:wrap;gap:10px'>{cells}</div>"


def filters_tab(*, clips, thresholds, thumbs, **_) -> str:
    ex = examples(clips)
    blocks = []
    for f in FILTERS:
        key, label, lo, hi = METRIC[f]
        bounds = _bounds(f, thresholds)
        fmt = (lambda v: f"{v:.1%}") if f == "text" else (lambda v: f"{v:.2f}" if abs(v) < 20 else f"{v:.0f}")

        def passes(v, f=f, key=key):
            return select_one(f, v, thresholds)

        n_fail = sum(not c[f"pass_{f}"] for c in clips)
        first = sum(c["first_fail"] == f for c in clips)
        rejected, kept = ex[f]
        blocks.append(viz.panel(
            f"{f}: {label}",
            viz.row(
                viz.panel("", histogram([c[key] for c in clips], lo, hi, bounds, passes, log=f in ("text", "motion"),
                                        fmt=(lambda v: f"{v:.0%}") if f == "text" else viz.compact)
                          + ("<p class='note'>Log scale.</p>" if f in ("text", "motion") else "")),
                viz.panel("", f"<h3>Rejected here first ({first})</h3>" + _sheet(rejected, key, thumbs, fmt)
                          + "<h3 style='margin-top:12px'>Kept</h3>" + _sheet(kept, key, thumbs, fmt)),
            ),
            f"{n_fail} of {len(clips)} clips fail this filter on its own; {first} of them passed every earlier filter.",
        ))
    return viz.page("Filters", "Each filter's score distribution with its thresholds, and clips it rejected next "
                               "to clips that were kept, spread over the score range.", *blocks)


def select_one(f: str, v: float, t) -> bool:
    return {
        "exposure": t.min_brightness <= v <= t.max_brightness,
        "blur": v >= t.min_sharpness,
        "text": v <= t.max_text,
        "aesthetic": v >= t.min_aesthetic,
        "motion": t.min_motion <= v <= t.max_motion,
    }[f]


def _caption_cards(clips: list[dict], sample: list[str], frames: dict) -> str:
    by_id = {c["id"]: c for c in clips}
    out = []
    for cid in sample:
        c = by_id[cid]
        out.append(
            f"<div class='panel' style='margin:0'>{img(strip_jpeg(frames[cid]), cid, fluid=True)}"
            f"<p style='margin:8px 0 4px'>{viz.esc(c['caption'])}</p>"
            f"<p class='note'>{viz.esc(SOURCES[c['source']].title)} · {c['start_s']:.1f}-{c['end_s']:.1f}s · "
            f"aesthetic {c['aesthetic']:.2f} · motion {c['motion']:.2f} · <code>{cid}</code></p></div>")
    return f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:14px'>{''.join(out)}</div>"


def captions_tab(*, clips, sample, frames, vlm, **_) -> str:
    return viz.page("Captions", f"Kept clips and their captions from {viz.esc(vlm)}, which sees the four sampled "
                                "frames of each clip. Round robin over films, highest aesthetic score first.",
                    _caption_cards(clips, sample, frames))


def card(*, clips, summaries, thresholds, sample, frames, vlm, **_) -> str:
    kept = [c for c in clips if c["keep"]]
    credits = "".join(f"<li>{viz.esc(s.title)}: {viz.esc(s.credit)}. {viz.esc(s.license)}.</li>" for s in SOURCES.values())
    return viz.card(
        "Blender open movie clips",
        f"{len(kept)} captioned clips of 2 to 10 seconds, curated from {len(clips)} candidates cut from five Blender "
        "open movies with shot detection, exposure, blur, text, aesthetic and motion filters.",
        ["video", "text-to-video", "captions", "CC BY", vlm.split("/")[-1]],
        viz.tiles([("Clips", str(len(kept)), f"{sum(c['duration_s'] for c in kept) / 60:.1f} min"),
                   ("Kept", f"{len(kept) / max(len(clips), 1):.0%}", f"of {len(clips)} candidates"),
                   ("Films", str(len({c['source'] for c in clips})))], hero=True),
        viz.row(viz.panel("Funnel", _funnel_chart(clips)),
                viz.panel("Thresholds", viz.code(json.dumps(asdict(thresholds), indent=2)))),
        viz.panel("Sample", _caption_cards(clips, sample[:8], frames)),
        viz.panel("Contents", "<ul><li><code>manifest.parquet</code>: one row per candidate clip, with its source, "
                  "time range, scores, filter results, keep flag, caption and the URI of its MP4</li>"
                  "<li><code>sample/</code>: a sample of kept clips and their frame strips</li>"
                  "<li><code>ATTRIBUTION.txt</code>, <code>thresholds.json</code></li></ul>"),
        viz.panel("Attribution", f"<ul>{credits}</ul>"),
        viz.panel("Load it", viz.code(
            "import pandas as pd\nfrom flyte.io import Dir\nfrom flyte.remote import Artifact\n\n"
            "dataset = Artifact.get(\"blender-open-movie-clips\").to_python(Dir)\n"
            "manifest = pd.read_parquet(f\"{dataset.download_sync()}/manifest.parquet\")\n"
            "train = manifest[manifest.keep]  # caption and clip_uri for every kept clip")),
    )
