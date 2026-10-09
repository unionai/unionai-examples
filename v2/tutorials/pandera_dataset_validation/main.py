# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1,<2.11",
#    "matplotlib",
#    "pandera[torch]==0.34.0",
#    "torch",
# ]
# main = "main"
# params = "sizes_mb=[8,32] repeats=2 sweep_size_mb=64 sweep_chunk_rows=[256,4096,65536]"
# ///
"""What does it cost to validate a whole dataset once, with pandera?

The companion benchmark (../pandera_validation_overhead) measures
validating every batch inside a training loop, which every consumer of the data
pays on every epoch. This one measures the alternative: validate the dataset once,
where it's produced, so downstream consumers can trust it without re-checking.

It builds four synthetic datasets shaped like common training data, each with a
TensorDictSchema that checks keys, dtypes, shapes, and values:

- continuous_control: TorchRL/D4RL-style HalfCheetah transitions (17-dim
  observations, 6-dim actions in [-1, 1], rewards, next observations, done flags)
- atari_frames: DQN-style stacks of four 84x84 uint8 frames, an action from the
  18-action Atari set, a clipped reward, and a done flag
- tabular_classification: dense features, a one-hot categorical column, an
  integer label, and a sample weight
- token_sequences: LM fine-tuning sequences of 512 token ids, an attention mask,
  and labels that use -100 as the ignore index

For each dataset and size, the whole dataset is held in memory as one TensorDict
and validated by streaming it through ``validate()`` in large chunks (64 MB by
default). A second sweep validates a 1 GB dataset at different chunk sizes, from
a training-sized 256 rows up to the whole dataset at once, to show why one pass
in large chunks is cheaper per row than checking every training batch.

Run on the cluster configured in ~/.flyte/config.yaml:

    uv run main.py

Run locally:

    uv run main.py --local --sizes-mb 8 32 128 --repeats 2 \
        --sweep-size-mb 64 --sweep-chunk-rows 256 4096 65536
"""

import argparse
import base64
import json
import statistics
import time
from pathlib import Path

import flyte

# {{docs-fragment image-and-env}}
image = (
    flyte.Image.from_debian_base(python_version=(3, 12), name="pandera-tensordict-dataset-bench")
    .with_pip_packages("torch==2.*", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("pandera[torch]==0.34.0", "matplotlib")
    .with_apt_packages("fonts-inter")
)

# torch threads must match the CPU request: os.cpu_count() in a pod reports the
# node's cores, and oversubscribing the limit makes everything crawl.
CPUS = 4

env = flyte.TaskEnvironment(
    name="pandera_tensordict_dataset_bench",
    image=image,
    resources=flyte.Resources(cpu=CPUS, memory="12Gi"),
)
# {{/docs-fragment image-and-env}}

MB = 1024 * 1024
KINDS = ("continuous_control", "atari_frames", "tabular_classification", "token_sequences")
LABELS = {
    "continuous_control": "continuous control",
    "atari_frames": "Atari frames",
    "tabular_classification": "tabular classification",
    "token_sequences": "token sequences",
}


# {{docs-fragment schemas}}
def build_schema(kind: str):
    """The schema for one dataset kind.

    These use the object-based TensorDictSchema API because several keys need
    custom checks, which TensorDictModel fields don't accept in pandera 0.34.
    """
    import torch
    import pandera.tensordict as pa
    from pandera import Check

    def t(dtype, shape, *checks):
        return pa.Tensor(dtype=dtype, shape=shape, checks=list(checks) or None)

    if kind == "continuous_control":
        # HalfCheetah: 17-dim observations, 6 continuous actions bounded to [-1, 1].
        keys = {
            "observation": t(torch.float32, (None, 17), Check.in_range(-100.0, 100.0)),
            "action": t(torch.float32, (None, 6), Check.in_range(-1.0, 1.0)),
            "reward": t(torch.float32, (None,), Check.in_range(-100.0, 100.0)),
            "next_observation": t(torch.float32, (None, 17), Check.in_range(-100.0, 100.0)),
            "done": t(torch.bool, (None,)),
            "terminated": t(torch.bool, (None,)),
        }
    elif kind == "atari_frames":
        # uint8 already bounds pixel values, so the pixel check catches blank frames
        # instead, a common symptom of a bad reset or a broken renderer.
        not_blank = Check(lambda x: x.flatten(2).amax(dim=-1).gt(0).all(dim=-1), name="no_blank_frames")
        keys = {
            "pixels": t(torch.uint8, (None, 4, 84, 84), not_blank),
            "action": t(torch.int64, (None,), Check.in_range(0, 17)),
            "reward": t(torch.float32, (None,), Check.in_range(-1.0, 1.0)),
            "done": t(torch.bool, (None,)),
        }
    elif kind == "tabular_classification":
        # A valid one-hot row is all zeros and ones and sums to exactly one.
        one_hot = Check(lambda x: ((x == 0) | (x == 1)).all(dim=-1) & (x.sum(dim=-1) == 1), name="one_hot")
        finite = Check(lambda x: torch.isfinite(x), name="finite")
        keys = {
            "features": t(torch.float32, (None, 32), finite),
            "category": t(torch.float32, (None, 12), one_hot),
            "label": t(torch.int64, (None,), Check.in_range(0, 9)),
            "sample_weight": t(torch.float32, (None,), Check.ge(0.0)),
        }
    elif kind == "token_sequences":
        vocab = 32_000
        # Labels are token ids, or -100 where the loss should ignore the position.
        label_ids = Check(lambda x: (x == -100) | ((x >= 0) & (x < vocab)), name="label_ids")
        keys = {
            "input_ids": t(torch.int64, (None, 512), Check.in_range(0, vocab - 1)),
            "attention_mask": t(torch.bool, (None, 512)),
            "labels": t(torch.int64, (None, 512), label_ids),
        }
    else:
        raise ValueError(f"unknown dataset kind: {kind}")
    return pa.TensorDictSchema(keys=keys, batch_size=(None,))
# {{/docs-fragment schemas}}


# {{docs-fragment scan-check}}
def corrupt_last_row(kind: str, td) -> None:
    """Break one value in the last row, so a schema that passes didn't scan it all."""
    if kind == "continuous_control":
        td["action"][-1, 0] = 2.0
    elif kind == "atari_frames":
        td["pixels"][-1, 3] = 0
    elif kind == "tabular_classification":
        td["category"][-1] = 0.0
    elif kind == "token_sequences":
        td["labels"][-1, 300] = -5


def assert_checks_scan_everything(kind: str, schema, chunk_rows: int) -> None:
    import pandera.tensordict as pa

    td = make_dataset(kind, 3 * chunk_rows + 7)
    validate_streamed(schema, td, chunk_rows)
    corrupt_last_row(kind, td)
    try:
        validate_streamed(schema, td, chunk_rows)
    except pa.SchemaError:
        return
    raise AssertionError(f"{kind}: schema didn't catch a corrupted value in the last row")
# {{/docs-fragment scan-check}}


def bytes_per_row(kind: str) -> int:
    return {
        "continuous_control": (17 + 6 + 1 + 17) * 4 + 2,
        "atari_frames": 4 * 84 * 84 + 8 + 4 + 1,
        "tabular_classification": (32 + 12) * 4 + 8 + 4,
        "token_sequences": 512 * 8 * 2 + 512,
    }[kind]


def make_dataset(kind: str, n: int, seed: int = 0):
    """A valid synthetic dataset of ``n`` rows, built in chunks to cap peak memory."""
    import torch
    from tensordict import TensorDict

    gen = torch.Generator().manual_seed(seed)

    def fill(empty: dict, step: int, make_chunk) -> dict:
        for start in range(0, n, step):
            stop = min(start + step, n)
            for key, value in make_chunk(stop - start).items():
                empty[key][start:stop] = value
        return empty

    if kind == "continuous_control":
        data = {
            "observation": torch.empty(n, 17), "action": torch.empty(n, 6), "reward": torch.empty(n),
            "next_observation": torch.empty(n, 17), "done": torch.empty(n, dtype=torch.bool),
            "terminated": torch.empty(n, dtype=torch.bool),
        }
        data = fill(data, 1_000_000, lambda k: {
            "observation": torch.randn(k, 17, generator=gen) * 3,
            "action": torch.rand(k, 6, generator=gen) * 2 - 1,
            "reward": torch.randn(k, generator=gen),
            "next_observation": torch.randn(k, 17, generator=gen) * 3,
            "done": torch.rand(k, generator=gen) < 0.001,
            "terminated": torch.rand(k, generator=gen) < 0.001,
        })
    elif kind == "atari_frames":
        data = {
            "pixels": torch.empty(n, 4, 84, 84, dtype=torch.uint8), "action": torch.empty(n, dtype=torch.int64),
            "reward": torch.empty(n), "done": torch.empty(n, dtype=torch.bool),
        }
        data = fill(data, 10_000, lambda k: {
            "pixels": torch.randint(0, 256, (k, 4, 84, 84), generator=gen, dtype=torch.uint8),
            "action": torch.randint(0, 18, (k,), generator=gen),
            "reward": torch.randint(-1, 2, (k,), generator=gen).float(),
            "done": torch.rand(k, generator=gen) < 0.001,
        })
    elif kind == "tabular_classification":
        data = {
            "features": torch.empty(n, 32), "category": torch.empty(n, 12), "label": torch.empty(n, dtype=torch.int64),
            "sample_weight": torch.empty(n),
        }

        def chunk(k):
            category = torch.nn.functional.one_hot(torch.randint(0, 12, (k,), generator=gen), 12).float()
            return {
                "features": torch.randn(k, 32, generator=gen),
                "category": category,
                "label": torch.randint(0, 10, (k,), generator=gen),
                "sample_weight": torch.rand(k, generator=gen),
            }

        data = fill(data, 1_000_000, chunk)
    elif kind == "token_sequences":
        data = {
            "input_ids": torch.empty(n, 512, dtype=torch.int64),
            "attention_mask": torch.empty(n, 512, dtype=torch.bool),
            "labels": torch.empty(n, 512, dtype=torch.int64),
        }

        def chunk(k):
            input_ids = torch.randint(0, 32_000, (k, 512), generator=gen)
            # Mask out the prompt: the first 128 positions don't contribute to the loss.
            labels = input_ids.clone()
            labels[:, :128] = -100
            return {"input_ids": input_ids, "attention_mask": torch.ones(k, 512, dtype=torch.bool), "labels": labels}

        data = fill(data, 20_000, chunk)
    else:
        raise ValueError(f"unknown dataset kind: {kind}")
    return TensorDict(data, batch_size=[n])


# {{docs-fragment validate-streamed}}
def validate_streamed(schema, td, chunk_rows: int) -> float:
    """Validate ``td`` in chunks of ``chunk_rows`` rows and return wall-clock seconds.

    Slicing a TensorDict returns views, so chunking adds no copies, and
    ``inplace=True`` skips the defensive clone ``validate()`` makes by default.
    """
    n = td.batch_size[0]
    start = time.perf_counter()
    for i in range(0, n, chunk_rows):
        schema.validate(td[i : i + chunk_rows], inplace=True)
    return time.perf_counter() - start
# {{/docs-fragment validate-streamed}}


# {{docs-fragment run-benchmark}}
@env.task
async def run_dataset_benchmark(
    sizes_mb: list[int],
    repeats: int,
    chunk_mb: int,
    sweep_size_mb: int,
    sweep_chunk_rows: list[int],
) -> str:
    """Time whole-dataset validation for every kind and size on this one pod."""
    import gc
    import os
    import platform

    import torch

    torch.set_num_threads(min(CPUS, os.cpu_count() or 1))

    size_rows = []
    for kind in KINDS:
        schema = build_schema(kind)
        bpr = bytes_per_row(kind)
        chunk_rows = max(1, chunk_mb * MB // bpr)
        # Doubles as a warmup, so first-call costs don't land on the smallest size.
        assert_checks_scan_everything(kind, schema, min(chunk_rows, 4096))
        for size_mb in sizes_mb:
            n = max(1, size_mb * MB // bpr)
            td = make_dataset(kind, n)
            runs = [validate_streamed(schema, td, chunk_rows) for _ in range(repeats)]
            seconds = statistics.median(runs)
            size_rows.append({
                "kind": kind,
                "size_mb": size_mb,
                "rows": n,
                "bytes": n * bpr,
                "chunk_rows": chunk_rows,
                "seconds": seconds,
                "seconds_runs": runs,
                "gb_per_s": n * bpr / 1e9 / seconds,
                "ns_per_row": 1e9 * seconds / n,
            })
            print(f"{kind:>24} {size_mb:>6} MB  {n:>11,} rows  {seconds:8.3f} s", flush=True)
            del td
            gc.collect()

    # Chunk-size sweep on one dataset: the same 1 GB validated 256 rows at a time
    # (like a training batch) up to all at once.
    sweep_kind = "continuous_control"
    schema = build_schema(sweep_kind)
    bpr = bytes_per_row(sweep_kind)
    n = sweep_size_mb * MB // bpr
    td = make_dataset(sweep_kind, n)
    sweep_rows = []
    for chunk_rows in [*sweep_chunk_rows, n]:
        runs = [validate_streamed(schema, td, min(chunk_rows, n)) for _ in range(repeats)]
        seconds = statistics.median(runs)
        sweep_rows.append({
            "chunk_rows": min(chunk_rows, n),
            "whole_dataset": chunk_rows >= n,
            "seconds": seconds,
            "seconds_runs": runs,
            "ns_per_row": 1e9 * seconds / n,
        })
        print(f"sweep chunk={min(chunk_rows, n):>11,} rows  {seconds:8.3f} s", flush=True)

    return json.dumps({
        "config": {
            "sizes_mb": sizes_mb, "repeats": repeats, "chunk_mb": chunk_mb,
            "sweep_kind": sweep_kind, "sweep_size_mb": sweep_size_mb, "sweep_rows": n,
            "bytes_per_row": {k: bytes_per_row(k) for k in KINDS},
        },
        "environment": {
            "torch": torch.__version__,
            "pandera": __import__("pandera").__version__,
            "tensordict": __import__("tensordict").__version__,
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "machine": platform.machine(),
        },
        "sizes": size_rows,
        "sweep": sweep_rows,
    }, indent=2)
# {{/docs-fragment run-benchmark}}


# Categorical slots 1-4 of the dataviz reference palette in fixed order, validated
# as a set in both modes. In light mode slots 3 and 4 sit under 3:1 contrast on the
# surface, so every series is also direct-labeled and the report has a table.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "surface": "#fcfcfb", "ink": "#0b0b0b",
        "ink_secondary": "#52514e", "ink_muted": "#8a897f", "grid": "#e6e5e0", "bar": "#2a78d6",
        "bar_muted": "#a9c8ee",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"], "surface": "#1a1a19", "ink": "#ffffff",
        "ink_secondary": "#c3c2b7", "ink_muted": "#8f8e86", "grid": "#333331", "bar": "#3987e5",
        "bar_muted": "#28466b",
    },
}


def _fmt_bytes(b: float) -> str:
    return f"{b / 1024**3:g} GB" if b >= 1024**3 else f"{b / MB:g} MB"


def _fmt_seconds(s: float) -> str:
    if s < 0.01:
        return f"{s * 1000:.1f} ms"
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s"


def _chunk_label(r: dict) -> str:
    return "whole dataset" if r["whole_dataset"] else f"{r['chunk_rows']:,} rows"


def render_plot(results: dict, out: Path, theme: str = "light") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    c = THEMES[theme]
    SURFACE, INK, INK_SECONDARY, INK_MUTED, GRID = c["surface"], c["ink"], c["ink_secondary"], c["ink_muted"], c["grid"]
    cfg, env_info = results["config"], results["environment"]
    sizes, sweep = results["sizes"], results["sweep"]

    largest_mb = max(cfg["sizes_mb"])
    at_largest = {r["kind"]: r for r in sizes if r["size_mb"] == largest_mb}
    slowest = max(at_largest.values(), key=lambda r: r["seconds"])
    per_batch = next(r for r in sweep if r["chunk_rows"] == min(x["chunk_rows"] for x in sweep))
    best = min(sweep, key=lambda r: r["seconds"])

    plt.rcParams.update({
        "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlepad": 12,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 8,
        "ytick.major.pad": 6,
    })
    fig, (ax_size, ax_chunk) = plt.subplots(
        1, 2, figsize=(12, 6.5), facecolor=SURFACE, gridspec_kw={"wspace": 0.3, "width_ratios": [1.35, 1]}
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.79, bottom=0.23)

    fig.text(0.07, 0.94,
             f"Validating {_fmt_bytes(slowest['bytes'])} of training data once takes at most "
             f"{_fmt_seconds(slowest['seconds'])}",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.07, 0.885,
             "Validation time grows linearly with dataset size. Paid once where the data is produced, "
             "every downstream consumer can skip it.",
             fontsize=11.5, color=INK_SECONDARY)

    for ax in (ax_size, ax_chunk):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)

    # Left: validation time vs dataset size, one line per dataset kind, log-log.
    ax_size.grid(axis="y", color=GRID, linewidth=0.8)
    label_positions = []
    for kind, color in zip(KINDS, c["series"]):
        rows = sorted((r for r in sizes if r["kind"] == kind), key=lambda r: r["size_mb"])
        xs = [r["bytes"] for r in rows]
        ys = [r["seconds"] for r in rows]
        ax_size.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                     markeredgewidth=2, label=LABELS[kind], solid_capstyle="round", zorder=3)
        label_positions.append((ys[-1], xs[-1], kind, color))
    ax_size.set_xscale("log", base=2)
    ax_size.set_yscale("log")
    # Room inside the axes, right of the last point, for the direct labels.
    all_bytes = [r["bytes"] for r in sizes]
    ax_size.set_xlim(min(all_bytes) / 1.5, max(all_bytes) * 9)
    ax_size.legend(frameon=False, loc="upper left", handlelength=1.6, labelcolor=INK_SECONDARY, fontsize=10)

    # Direct labels at the right end of each line, spaced at least one line of text
    # apart in screen space (data units on a log axis would bunch them unevenly).
    min_gap_px = 14 * fig.dpi / 72
    to_px, to_data = ax_size.transData, ax_size.transData.inverted()
    placed_px = None
    for y, x, kind, _ in sorted(label_positions):
        _, py = to_px.transform((x, y))
        if placed_px is not None:
            py = max(py, placed_px + min_gap_px)
        placed_px = py
        _, label_y = to_data.transform((0, py))
        ax_size.annotate(f"{_fmt_seconds(y)}  {LABELS[kind]}", xy=(x, y), xytext=(x * 1.2, label_y),
                         textcoords="data", va="center", color=INK, fontsize=10)
    xticks = sorted({r["bytes"] for r in sizes if r["kind"] == KINDS[0]})
    ax_size.set_xticks(xticks, [_fmt_bytes(mb * MB) for mb in sorted(cfg["sizes_mb"])])
    ax_size.xaxis.set_minor_locator(NullLocator())
    ax_size.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax_size.yaxis.set_minor_locator(NullLocator())
    ax_size.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1000:g} ms" if v < 1 else f"{v:g} s"))
    ax_size.set_xlabel("dataset size in memory")
    ax_size.set_title("Time to validate the whole dataset", loc="left")

    # Right: the same data validated in different chunk sizes. Horizontal bars, so
    # the chunk-size labels read naturally; the training-batch-sized chunk is muted.
    ax_chunk.grid(axis="x", color=GRID, linewidth=0.8)
    labels, values, colors = [], [], []
    for r in sweep:
        labels.append(_chunk_label(r))
        values.append(r["seconds"])
        colors.append(c["bar_muted"] if r is per_batch else c["bar"])
    ypos = list(range(len(sweep)))[::-1]
    ax_chunk.barh(ypos, values, height=0.62, color=colors, zorder=2)
    for y, v in zip(ypos, values):
        ax_chunk.annotate(_fmt_seconds(v), (v, y), xytext=(6, 0), textcoords="offset points", va="center",
                          color=INK, fontsize=10)
    ax_chunk.set_yticks(ypos, labels)
    ax_chunk.set_xlim(0, max(values) * 1.22)
    ax_chunk.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1000:g} ms" if v < 1 else f"{v:g} s"))
    ax_chunk.set_xlabel("wall-clock time")
    ax_chunk.set_title(f"Same {_fmt_bytes(cfg['sweep_size_mb'] * MB)}, validated in chunks of", loc="left")
    ax_chunk.annotate(
        f"{per_batch['seconds'] / best['seconds']:.0f}x slower in\ntraining-batch-sized chunks",
        (per_batch["seconds"], ypos[sweep.index(per_batch)]), xytext=(0, -26), textcoords="offset points",
        ha="right", va="top", color=INK_SECONDARY, fontsize=9.5,
    )

    fig.text(
        0.07, 0.025,
        f"Median of {cfg['repeats']} runs. Each dataset is one in-memory TensorDict streamed through validate(inplace=True) "
        f"in {cfg['chunk_mb']} MB chunks.\nEvery key gets dtype and shape checks, and every non-boolean key a value check. "
        f"Chunk sweep: {LABELS[cfg['sweep_kind']]}, {cfg['sweep_rows']:,} rows.\n"
        f"{env_info['torch_threads']} CPU threads, {env_info['machine']}, pandera {env_info['pandera']}, "
        f"torch {env_info['torch']}, tensordict {env_info['tensordict']}.",
        color=INK_MUTED, fontsize=9, linespacing=1.5, va="bottom",
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def report_html(data: dict, light_png: bytes, dark_png: bytes) -> str:
    """The report page: chart and tables, with a light/dark toggle.

    The toggle is a pair of radio buttons styled with :has(), so it works without
    JavaScript; a two-line script only preselects dark when the viewer's OS is dark.
    """
    light, dark = THEMES["light"], THEMES["dark"]

    def css_vars(t: dict) -> str:
        return (f"--surface:{t['surface']};--ink:{t['ink']};--ink-2:{t['ink_secondary']};"
                f"--ink-3:{t['ink_muted']};--grid:{t['grid']};--accent:{t['series'][0]};")

    size_html = "".join(
        f"<tr><td>{LABELS[r['kind']]}</td><td>{_fmt_bytes(r['bytes'])}</td><td>{r['rows']:,}</td>"
        f"<td>{r['seconds']:.3f}</td><td>{r['gb_per_s']:.2f}</td><td>{r['ns_per_row']:.1f}</td></tr>"
        for r in data["sizes"]
    )
    sweep_html = "".join(
        f"<tr><td>{_chunk_label(r)}</td><td>{r['seconds']:.3f}</td><td>{r['ns_per_row']:.1f}</td></tr>"
        for r in data["sweep"]
    )
    light_b64 = base64.b64encode(light_png).decode()
    dark_b64 = base64.b64encode(dark_png).decode()
    return f"""
<style>
  .bench {{ {css_vars(light)} background:var(--surface); color:var(--ink); padding:24px 28px 32px;
           font-family:Inter,"Helvetica Neue",Arial,sans-serif; border-radius:12px; }}
  .bench:has(#bench-dark:checked) {{ {css_vars(dark)} }}
  .bench header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }}
  .bench h2 {{ margin:0; font-size:20px; font-weight:700; }}
  .bench h3 {{ margin:24px 0 8px; font-size:14px; font-weight:600; color:var(--ink-2); }}
  .bench .toggle {{ display:inline-flex; border:1px solid var(--grid); border-radius:999px; padding:3px; }}
  .bench .toggle input {{ position:absolute; opacity:0; pointer-events:none; }}
  .bench .toggle label {{ padding:5px 14px; border-radius:999px; font-size:13px; color:var(--ink-2); cursor:pointer; }}
  .bench #bench-light:checked + label, .bench #bench-dark:checked + label {{ background:var(--grid); color:var(--ink); }}
  .bench .toggle input:focus-visible + label {{ outline:2px solid var(--accent); outline-offset:2px; }}
  .bench img {{ width:100%; height:auto; display:block; margin:20px 0 8px; border-radius:8px; }}
  .bench .chart-dark, .bench:has(#bench-dark:checked) .chart-light {{ display:none; }}
  .bench:has(#bench-dark:checked) .chart-dark {{ display:block; }}
  .bench table {{ width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }}
  .bench th {{ text-align:right; font-weight:600; color:var(--ink-2); padding:8px 10px; border-bottom:1px solid var(--grid); }}
  .bench td {{ text-align:right; padding:8px 10px; border-bottom:1px solid var(--grid); }}
  .bench th:first-child, .bench td:first-child {{ text-align:left; }}
</style>
<div class="bench">
  <header>
    <h2>pandera whole-dataset validation</h2>
    <div class="toggle" role="radiogroup" aria-label="Color theme">
      <input type="radio" name="bench-theme" id="bench-light" checked><label for="bench-light">Light</label>
      <input type="radio" name="bench-theme" id="bench-dark"><label for="bench-dark">Dark</label>
    </div>
  </header>
  <img class="chart-light" alt="Validation time vs dataset size, light theme" src="data:image/png;base64,{light_b64}">
  <img class="chart-dark" alt="Validation time vs dataset size, dark theme" src="data:image/png;base64,{dark_b64}">
  <h3>Validation time by dataset size</h3>
  <table>
    <tr><th>dataset</th><th>size</th><th>rows</th><th>seconds</th><th>GB/s</th><th>ns/row</th></tr>
    {size_html}
  </table>
  <h3>Chunk-size sweep ({LABELS[data["config"]["sweep_kind"]]}, {data["config"]["sweep_rows"]:,} rows)</h3>
  <table>
    <tr><th>chunk</th><th>seconds</th><th>ns/row</th></tr>
    {sweep_html}
  </table>
</div>
<script>
  if (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches)
    document.getElementById("bench-dark").checked = true;
</script>
"""


# {{docs-fragment report-and-main}}
@env.task(report=True)
async def plot(results: str) -> None:
    """Render the chart in both themes into the run's report tab."""
    data = json.loads(results)
    pngs = {}
    for theme in ("light", "dark"):
        out = Path(f"dataset-validation-{theme}.png")
        render_plot(data, out, theme=theme)
        pngs[theme] = out.read_bytes()
    await flyte.report.replace.aio(report_html(data, pngs["light"], pngs["dark"]))
    await flyte.report.flush.aio()


@env.task
async def main(
    sizes_mb: list[int] = [16, 64, 256, 1024, 4096],
    repeats: int = 3,
    chunk_mb: int = 64,
    sweep_size_mb: int = 1024,
    sweep_chunk_rows: list[int] = [256, 4_096, 65_536, 1_048_576],
) -> str:
    results = await run_dataset_benchmark(sizes_mb, repeats, chunk_mb, sweep_size_mb, sweep_chunk_rows)
    await plot(results)
    return results
# {{/docs-fragment report-and-main}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run in-process instead of on the cluster")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--sweep-size-mb", type=int, default=1024)
    parser.add_argument("--sweep-chunk-rows", type=int, nargs="+", default=[256, 4_096, 65_536, 1_048_576])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    kwargs = dict(
        sizes_mb=args.sizes_mb, repeats=args.repeats, chunk_mb=args.chunk_mb,
        sweep_size_mb=args.sweep_size_mb, sweep_chunk_rows=args.sweep_chunk_rows,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.local:
        flyte.init()
        run = flyte.with_runcontext(mode="local").run(main, **kwargs)
    else:
        flyte.init_from_config()
        run = flyte.run(main, **kwargs)
        print(f"Run: {run.url}")
        run.wait()
    results = run.outputs()

    if not isinstance(results, str):  # some SDK versions wrap single outputs in a tuple
        results = results[0]
    (args.out / "results.json").write_text(results)
    data = json.loads(results)
    render_plot(data, args.out / "dataset-validation.png")
    render_plot(data, args.out / "dataset-validation-dark.png", theme="dark")
    print(f"Wrote {args.out}/results.json and the light and dark charts")
