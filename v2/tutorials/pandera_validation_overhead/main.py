# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1,<2.11",
#    "matplotlib",
#    "pandera[torch]==0.34.0",
#    "torch",
# ]
# main = "main"
# params = "lengths=[100,300] repeats=2 widths=[64,256] sweep_repeats=2 sweep_target_seconds=1.0"
# ///
"""How much does validating every training batch with pandera cost?

Trains a small policy network on TensorDict batches of RL transitions, where a
fraction of the batches are corrupted in the three ways the blog post describes
(float64 observations, an out-of-range action, a NaN reward). Each training
length is run in two modes on the same pod:

- without pandera: the loop skips corrupt batches using a precomputed mask, which
  is the cheapest possible skip and the baseline we compare against.
- with pandera: the loop calls ``Transition.validate`` on every batch and skips
  the ones that raise ``SchemaError``.

Both modes train on exactly the same batches, so the difference in wall-clock
time is the cost of validation. The script asserts that pandera skips the same
batches as the mask.

A second task, run_step_size_sweep, holds the batch (and so the validation
work) fixed and grows the model, to show how validation's share of the step
falls as the step gets more expensive.

Run on the cluster configured in ~/.flyte/config.yaml:

    uv run main.py

Run locally:

    uv run main.py --local --lengths 100 300 --repeats 2 \
        --widths 64 256 --sweep-repeats 2 --sweep-target-seconds 1
"""

import argparse
import base64
import json
import random
import statistics
import time
from pathlib import Path

import flyte

# {{docs-fragment image-and-env}}
image = (
    flyte.Image.from_debian_base(python_version=(3, 12), name="pandera-tensordict-bench")
    .with_pip_packages("torch==2.*", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("pandera[torch]==0.34.0", "matplotlib")
    .with_apt_packages("fonts-inter")
)

# torch threads must match the CPU request: os.cpu_count() in a pod reports the
# node's cores, and oversubscribing a 4-CPU limit makes every step crawl.
CPUS = 4

# The limit sits above the thread count on purpose. With limit == threads, a
# matmul-heavy step uses up the CFS quota and gets throttled, and the throttling
# pattern shifts when validation runs between steps, which skews the comparison.
env = flyte.TaskEnvironment(
    name="pandera_tensordict_bench",
    image=image,
    resources=flyte.Resources(cpu=(CPUS, 2 * CPUS), memory="8Gi"),
)
# {{/docs-fragment image-and-env}}


def cpu_throttled_ms() -> float | None:
    """Time this cgroup has spent CPU-throttled, from cgroup v2's cpu.stat."""
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            key, value = line.split()
            if key == "throttled_usec":
                return int(value) / 1000
    except OSError:
        pass
    return None

N_ACTIONS = 4
OBS_BOUND = 5.0
CORRUPTIONS = ("float64_observation", "action_out_of_range", "nan_reward")


# {{docs-fragment schema}}
def build_schema(obs_dim: int):
    import torch
    import pandera.tensordict as pa

    class Transition(pa.TensorDictModel):
        observation: torch.float32 = pa.Field(shape=(None, obs_dim), ge=-OBS_BOUND, le=OBS_BOUND)
        action: torch.int64 = pa.Field(shape=(None,), isin=list(range(N_ACTIONS)))
        reward: torch.float32 = pa.Field(shape=(None,), ge=-1.0, le=1.0)
        done: torch.bool = pa.Field(shape=(None,))

        class Config:
            batch_size = (None,)

    return Transition
# {{/docs-fragment schema}}


# {{docs-fragment corrupt-batches}}
def make_pool(pool_size: int, batch_size: int, obs_dim: int, corrupt_rate: float, seed: int):
    """A fixed pool of batches the training loop cycles through.

    Generating batches up front keeps data generation out of the timed loop, and
    a pool (rather than one batch per step) keeps memory flat at long lengths.
    """
    import torch
    from tensordict import TensorDict

    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    pool = []
    for _ in range(pool_size):
        observation = torch.randn(batch_size, obs_dim, generator=gen).clamp(-OBS_BOUND, OBS_BOUND)
        action = torch.randint(0, N_ACTIONS, (batch_size,), generator=gen)
        reward = torch.rand(batch_size, generator=gen) * 2 - 1
        done = torch.rand(batch_size, generator=gen) < 0.01

        corruption = rng.choice(CORRUPTIONS) if rng.random() < corrupt_rate else None
        if corruption == "float64_observation":
            observation = observation.double()
        elif corruption == "action_out_of_range":
            action[rng.randrange(batch_size)] = N_ACTIONS
        elif corruption == "nan_reward":
            reward[rng.randrange(batch_size)] = float("nan")

        td = TensorDict(
            {"observation": observation, "action": action, "reward": reward, "done": done},
            batch_size=[batch_size],
        )
        pool.append((td, corruption is None))
    return pool
# {{/docs-fragment corrupt-batches}}


# {{docs-fragment train}}
def train(n_steps: int, pool: list, obs_dim: int, hidden: int, schema=None, seed: int = 0) -> dict:
    """Run ``n_steps`` of a REINFORCE-style update, skipping invalid batches."""
    import torch
    import pandera.tensordict as pa

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_ACTIONS),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trained, skipped, skipped_steps, validate_s = 0, 0, [], 0.0
    start = time.perf_counter()
    for step in range(n_steps):
        td, is_valid = pool[step % len(pool)]

        if schema is not None:
            t0 = time.perf_counter()
            try:
                schema.validate(td)
                is_valid = True
            except pa.SchemaError:
                is_valid = False
            validate_s += time.perf_counter() - t0

        if not is_valid:
            skipped += 1
            skipped_steps.append(step)
            continue

        logits = model(td["observation"])
        log_prob = torch.log_softmax(logits, dim=-1).gather(1, td["action"].unsqueeze(1)).squeeze(1)
        loss = -(log_prob * td["reward"]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trained += 1

    return {
        "seconds": time.perf_counter() - start,
        "validate_seconds": validate_s,
        "trained": trained,
        "skipped": skipped,
        "skipped_steps": skipped_steps,
    }
# {{/docs-fragment train}}


# {{docs-fragment run-benchmark}}
@env.task
async def run_benchmark(
    lengths: list[int],
    repeats: int,
    batch_size: int,
    obs_dim: int,
    hidden: int,
    corrupt_rate: float,
    pool_size: int,
) -> str:
    """Time both modes at every training length on this one pod."""
    import os
    import platform

    import torch

    torch.set_num_threads(min(CPUS, os.cpu_count() or 1))
    schema = build_schema(obs_dim)
    pool = make_pool(pool_size, batch_size, obs_dim, corrupt_rate, seed=0)

    # Warm up both paths so one-time costs (allocator, autograd, pandera's schema
    # build) don't land on the shortest run.
    train(50, pool, obs_dim, hidden)
    train(50, pool, obs_dim, hidden, schema=schema)

    rows = []
    for n_steps in lengths:
        base_runs, pandera_runs = [], []
        for r in range(repeats):
            # Alternate which mode goes first so drift on the node doesn't
            # systematically favor one of them.
            order = [("base", None), ("pandera", schema)]
            if r % 2:
                order.reverse()
            for mode, s in order:
                result = train(n_steps, pool, obs_dim, hidden, schema=s, seed=r)
                (base_runs if mode == "base" else pandera_runs).append(result)

        for b, p in zip(base_runs, pandera_runs):
            assert b["skipped_steps"] == p["skipped_steps"], "pandera skipped different batches than the mask"

        base_s = [x["seconds"] for x in base_runs]
        pandera_s = [x["seconds"] for x in pandera_runs]
        overhead = [100 * (p - b) / b for b, p in zip(base_s, pandera_s)]
        validated = n_steps * repeats
        row = {
            "steps": n_steps,
            "trained_batches": base_runs[0]["trained"],
            "skipped_batches": base_runs[0]["skipped"],
            "without_pandera_s": statistics.median(base_s),
            "without_pandera_s_runs": base_s,
            "with_pandera_s": statistics.median(pandera_s),
            "with_pandera_s_runs": pandera_s,
            "overhead_pct": statistics.median(overhead),
            "overhead_pct_runs": overhead,
            "validate_ms_per_batch": 1000 * sum(x["validate_seconds"] for x in pandera_runs) / validated,
            "train_ms_per_batch": 1000 * sum(base_s) / (base_runs[0]["trained"] * repeats),
        }
        rows.append(row)
        print(
            f"{n_steps:>6} steps  without={row['without_pandera_s']:.2f}s  "
            f"with={row['with_pandera_s']:.2f}s  overhead={row['overhead_pct']:.1f}%",
            flush=True,
        )

    results = {
        "config": {
            "lengths": lengths,
            "repeats": repeats,
            "batch_size": batch_size,
            "obs_dim": obs_dim,
            "hidden": hidden,
            "corrupt_rate": corrupt_rate,
            "pool_size": pool_size,
            "pool_corrupt_batches": sum(not ok for _, ok in pool),
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
        "rows": rows,
    }
    # Returned inline as a string rather than a File, so the driver can read it
    # from run.outputs() without credentials for the cluster's bucket.
    return json.dumps(results, indent=2)
# {{/docs-fragment run-benchmark}}


def mlp_params(obs_dim: int, hidden: int) -> int:
    return obs_dim * hidden + hidden + hidden * hidden + hidden + hidden * N_ACTIONS + N_ACTIONS


# {{docs-fragment step-size-sweep}}
@env.task
async def run_step_size_sweep(
    widths: list[int],
    repeats: int,
    batch_size: int,
    obs_dim: int,
    corrupt_rate: float,
    pool_size: int,
    target_seconds: float,
) -> str:
    """Hold the batch (and so the validation work) fixed and grow the model.

    Every width validates the same 256-row batches against the same schema, so
    pandera's cost per step stays constant while the forward/backward pass gets
    more expensive. Each width runs enough steps to take about ``target_seconds``.
    """
    import os
    import platform

    import torch

    torch.set_num_threads(min(CPUS, os.cpu_count() or 1))
    schema = build_schema(obs_dim)
    pool = make_pool(pool_size, batch_size, obs_dim, corrupt_rate, seed=0)
    train(50, pool, obs_dim, widths[0])
    train(50, pool, obs_dim, widths[0], schema=schema)

    rows = []
    for hidden in widths:
        probe = train(20, pool, obs_dim, hidden)
        step_s = probe["seconds"] / max(probe["trained"], 1)
        n_steps = int(min(3_000, max(50, target_seconds / step_s)))

        base_runs, pandera_runs = [], []
        for r in range(repeats):
            order = [("base", None), ("pandera", schema)]
            if r % 2:
                order.reverse()
            for mode, sch in order:
                before = cpu_throttled_ms()
                result = train(n_steps, pool, obs_dim, hidden, schema=sch, seed=r)
                after = cpu_throttled_ms()
                result["throttled_ms"] = after - before if before is not None and after is not None else None
                (base_runs if mode == "base" else pandera_runs).append(result)
        for b, p in zip(base_runs, pandera_runs):
            assert b["skipped_steps"] == p["skipped_steps"], "pandera skipped different batches than the mask"

        base_s = [x["seconds"] for x in base_runs]
        pandera_s = [x["seconds"] for x in pandera_runs]
        overhead = [100 * (p - b) / b for b, p in zip(base_s, pandera_s)]
        trained = base_runs[0]["trained"]
        row = {
            "hidden": hidden,
            "params": mlp_params(obs_dim, hidden),
            "steps": n_steps,
            "without_pandera_s": statistics.median(base_s),
            "without_pandera_s_runs": base_s,
            "with_pandera_s": statistics.median(pandera_s),
            "with_pandera_s_runs": pandera_s,
            "overhead_pct": statistics.median(overhead),
            "overhead_pct_runs": overhead,
            "validate_ms_per_batch": 1000 * sum(x["validate_seconds"] for x in pandera_runs) / (n_steps * repeats),
            "train_ms_per_step": 1000 * statistics.median(base_s) / trained,
            "throttled_ms_without": [x["throttled_ms"] for x in base_runs],
            "throttled_ms_with": [x["throttled_ms"] for x in pandera_runs],
        }
        rows.append(row)
        print(
            f"hidden={hidden:>5}  params={row['params']:>11,}  steps={n_steps:>5}  "
            f"train={row['train_ms_per_step']:8.2f} ms/step  validate={row['validate_ms_per_batch']:.3f} ms  "
            f"overhead={row['overhead_pct']:.1f}%",
            flush=True,
        )

    return json.dumps({
        "config": {
            "widths": widths, "repeats": repeats, "batch_size": batch_size, "obs_dim": obs_dim,
            "corrupt_rate": corrupt_rate, "pool_size": pool_size, "target_seconds": target_seconds,
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
        "rows": rows,
    }, indent=2)
# {{/docs-fragment step-size-sweep}}


# Categorical slots 1 and 2 of the dataviz reference palette (validated as a pair
# in both modes), with each mode's surface, text, and grid inks. The dark steps are
# chosen for the dark surface, not inverted from the light ones.
THEMES = {
    "light": {
        "without": "#2a78d6", "with": "#eb6834", "surface": "#fcfcfb", "ink": "#0b0b0b",
        "ink_secondary": "#52514e", "ink_muted": "#8a897f", "grid": "#e6e5e0",
        "range": "#f5c3ad",  # a light step of "with" for the min/max ranges
    },
    "dark": {
        "without": "#3987e5", "with": "#d95926", "surface": "#1a1a19", "ink": "#ffffff",
        "ink_secondary": "#c3c2b7", "ink_muted": "#8f8e86", "grid": "#333331",
        "range": "#5c3324",  # a dim step of "with" that recedes on the dark surface
    },
}


def render_plot(results: dict, out: Path, theme: str = "light") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    c = THEMES[theme]
    WITHOUT_COLOR, WITH_COLOR, SURFACE, INK = c["without"], c["with"], c["surface"], c["ink"]
    INK_SECONDARY, INK_MUTED, GRID, RANGE = c["ink_secondary"], c["ink_muted"], c["grid"], c["range"]

    rows = results["rows"]
    cfg = results["config"]
    env_info = results["environment"]
    steps = [r["steps"] for r in rows]
    pct = [r["overhead_pct"] for r in rows]
    overall = statistics.median(pct)
    validate_ms = statistics.mean(r["validate_ms_per_batch"] for r in rows)
    train_ms = statistics.mean(r["train_ms_per_batch"] for r in rows)

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
    fig, (ax_time, ax_pct) = plt.subplots(
        1, 2, figsize=(12, 6.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.28}
    )
    fig.subplots_adjust(left=0.07, right=0.92, top=0.79, bottom=0.2)

    fig.text(0.07, 0.94, f"Validating every batch adds about {overall:.0f}% to training time",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.07, 0.885,
             f"pandera spends {validate_ms:.2f} ms checking each {cfg['batch_size']}-row TensorDict against a "
             f"{train_ms:.2f} ms training step, so the share stays flat as runs get longer.",
             fontsize=11.5, color=INK_SECONDARY)

    for ax in (ax_time, ax_pct):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # Left: wall-clock time. Steps span two orders of magnitude, so log-log
    # lines, with the series named at their right ends as well as in the legend.
    series = (
        ([r["without_pandera_s"] for r in rows], WITHOUT_COLOR, "without pandera"),
        ([r["with_pandera_s"] for r in rows], WITH_COLOR, "with pandera"),
    )
    for values, color, label in series:
        ax_time.plot(steps, values, color=color, linewidth=2, marker="o", markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=2, label=label, solid_capstyle="round", zorder=3)
    ax_time.annotate(f"{series[1][0][-1]:.1f} s", (steps[-1], series[1][0][-1]), xytext=(10, 4),
                     textcoords="offset points", va="bottom", color=INK, fontsize=10.5)
    ax_time.annotate(f"{series[0][0][-1]:.1f} s", (steps[-1], series[0][0][-1]), xytext=(10, -4),
                     textcoords="offset points", va="top", color=INK, fontsize=10.5)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xticks(steps, [f"{s:,}" for s in steps])
    ax_time.xaxis.set_minor_locator(NullLocator())
    ax_time.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_time.yaxis.set_minor_locator(NullLocator())
    ax_time.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax_time.set_xlim(steps[0] / 1.4, steps[-1] * 2.2)
    ax_time.set_xlabel("training steps")
    ax_time.set_title("Wall-clock time", loc="left")
    ax_time.legend(frameon=False, loc="upper left", handlelength=1.6, labelcolor=INK_SECONDARY)

    # Right: overhead per length as a dot (median) on a min/max range line, with
    # the median across all lengths as a reference.
    x = list(range(len(rows)))
    for i, r in zip(x, rows):
        ax_pct.plot([i, i], [min(r["overhead_pct_runs"]), max(r["overhead_pct_runs"])], color=RANGE,
                    linewidth=6, solid_capstyle="round", zorder=2)
    ax_pct.scatter(x, pct, s=110, color=WITH_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
    for i, v in zip(x, pct):
        # A surface-colored pad lets the label sit cleanly over the reference line.
        ax_pct.annotate(f"{v:.1f}%", (i, v), xytext=(11, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=10.5,
                        bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})
    ax_pct.axhline(overall, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    # Label the reference line just outside the plot area, clear of the data.
    ax_pct.annotate(f"median\n{overall:.1f}%", (len(rows) - 0.05, overall), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", color=INK_SECONDARY,
                    fontsize=9.5, linespacing=1.2, annotation_clip=False)
    top = max(max(r["overhead_pct_runs"]) for r in rows)
    ax_pct.set_ylim(0, max(top * 1.15, 1))
    ax_pct.set_xlim(-0.5, len(rows) - 0.05)  # extra room on the right for the last value label
    ax_pct.set_xticks(x, [f"{s:,}" for s in steps])
    ax_pct.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax_pct.set_xlabel("training steps")
    ax_pct.set_title("Added runtime vs. without pandera", loc="left")

    fig.text(
        0.07, 0.025,
        f"Median of {cfg['repeats']} runs per length; the shaded range spans the fastest and slowest run. "
        f"MLP policy (hidden {cfg['hidden']}, obs dim {cfg['obs_dim']}), batch {cfg['batch_size']}, "
        f"{cfg['corrupt_rate']:.0%} corrupt batches skipped in both modes.\n"
        f"{env_info['torch_threads']} CPU threads, {env_info['machine']}, pandera {env_info['pandera']}, "
        f"torch {env_info['torch']}, tensordict {env_info['tensordict']}.",
        color=INK_MUTED, fontsize=9, linespacing=1.5, va="bottom",
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def render_step_plot(sweep: dict, out: Path, theme: str = "light") -> None:
    """Added runtime per model size, against what validation-cost / step-time predicts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    c = THEMES[theme]
    WITH_COLOR, SURFACE, INK = c["with"], c["surface"], c["ink"]
    INK_SECONDARY, INK_MUTED, GRID = c["ink_secondary"], c["ink_muted"], c["grid"]

    rows = sweep["rows"]
    cfg, env_info = sweep["config"], sweep["environment"]
    pct = [r["overhead_pct"] for r in rows]
    x = list(range(len(rows)))

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
    fig, ax = plt.subplots(figsize=(12, 6.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.25)

    # Both halves of the ratio are timed directly: validate() around each call, and the
    # step from the run without pandera. The end-to-end difference between the two modes
    # (overhead_pct) agrees where validation is a big share of the step, but past a few
    # percent it's smaller than the run-to-run noise on a shared node and lands on both
    # sides of zero, so it stays in the JSON and the README rather than on the chart.
    share = [100 * r["validate_ms_per_batch"] / r["train_ms_per_step"] for r in rows]
    first, last = rows[0], rows[-1]
    fig.text(0.07, 0.945, "The bigger the training step, the smaller pandera's share",
             fontsize=18, fontweight="bold", color=INK)
    v_lo = min(r["validate_ms_per_batch"] for r in rows)
    v_hi = max(r["validate_ms_per_batch"] for r in rows)
    fig.text(0.07, 0.905,
             f"Checking a {cfg['batch_size']}-row batch took {v_lo:.2f}–{v_hi:.2f} ms while the training step grew "
             f"{last['train_ms_per_step'] / first['train_ms_per_step']:.0f}x, so validation went from\n"
             f"{share[0]:.0f}% of a {first['train_ms_per_step']:.1f} ms step to {share[-1]:.1f}% of a "
             f"{last['train_ms_per_step']:.0f} ms step.",
             fontsize=11.5, color=INK_SECONDARY, va="top", linespacing=1.45)

    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.plot(x, share, color=WITH_COLOR, linewidth=1.6, zorder=2)
    ax.scatter(x, share, s=110, color=WITH_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
    for i, v in zip(x, share):
        ax.annotate(f"{v:.1f}%", (i, v), xytext=(11, 6), textcoords="offset points", va="bottom",
                    color=INK, fontsize=10.5,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})

    ax.set_xticks(x, [f"{_fmt_params(r['params'])} params\n{r['train_ms_per_step']:.1f} ms/step" for r in rows])
    ax.set_xlim(-0.5, len(rows) - 0.3)
    ax.set_ylim(0, max(share) * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("MLP policy size and training step time without validation", labelpad=10)
    ax.set_title("Validation time as a share of the training step", loc="left")

    widths = ", ".join(str(w) for w in cfg["widths"])
    fig.text(
        0.07, 0.025,
        f"Validation ms per batch (timed around each validate() call) ÷ median training ms per step without "
        f"pandera, {cfg['repeats']} runs per size. Batch {cfg['batch_size']},\nobs dim {cfg['obs_dim']}, hidden "
        f"widths {widths}, {cfg['corrupt_rate']:.0%} corrupt batches skipped. "
        f"{env_info['torch_threads']} CPU threads, {env_info['machine']}, pandera "
        f"{env_info['pandera']}, torch {env_info['torch']}, tensordict {env_info['tensordict']}.",
        color=INK_MUTED, fontsize=9, linespacing=1.5, va="bottom",
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def report_html(data: dict, light_png: bytes, dark_png: bytes, sweep: dict | None = None,
                step_pngs: dict | None = None) -> str:
    """The report page: chart and table, with a light/dark toggle.

    The toggle is a pair of radio buttons styled with :has(), so it works without
    JavaScript; a two-line script only preselects dark when the viewer's OS is dark.
    """
    light, dark = THEMES["light"], THEMES["dark"]

    def css_vars(t: dict) -> str:
        return (f"--surface:{t['surface']};--ink:{t['ink']};--ink-2:{t['ink_secondary']};"
                f"--ink-3:{t['ink_muted']};--grid:{t['grid']};--accent:{t['with']};")

    rows_html = "".join(
        f"<tr><td>{r['steps']:,}</td><td>{r['without_pandera_s']:.2f}</td><td>{r['with_pandera_s']:.2f}</td>"
        f"<td>{r['overhead_pct']:.1f}%</td><td>{min(r['overhead_pct_runs']):.1f}–{max(r['overhead_pct_runs']):.1f}%</td>"
        f"<td>{r['validate_ms_per_batch']:.3f}</td><td>{r['train_ms_per_batch']:.3f}</td></tr>"
        for r in data["rows"]
    )
    light_b64 = base64.b64encode(light_png).decode()
    dark_b64 = base64.b64encode(dark_png).decode()
    sweep_html = ""
    if sweep is not None and step_pngs is not None:
        sweep_rows = "".join(
            f"<tr><td>{r['hidden']:,}</td><td>{r['params']:,}</td><td>{r['steps']:,}</td>"
            f"<td>{r['train_ms_per_step']:.2f}</td><td>{r['validate_ms_per_batch']:.3f}</td>"
            f"<td>{r['overhead_pct']:.1f}%</td></tr>"
            for r in sweep["rows"]
        )
        sl = base64.b64encode(step_pngs["light"]).decode()
        sd = base64.b64encode(step_pngs["dark"]).decode()
        sweep_html = f"""
  <h3>Added runtime as the training step grows</h3>
  <img class="chart-light" alt="Added runtime by model size, light theme" src="data:image/png;base64,{sl}">
  <img class="chart-dark" alt="Added runtime by model size, dark theme" src="data:image/png;base64,{sd}">
  <table>
    <tr><th>hidden</th><th>params</th><th>steps</th><th>train ms/step</th><th>validate ms/batch</th><th>overhead</th></tr>
    {sweep_rows}
  </table>"""
    return f"""
<style>
  .bench {{ {css_vars(light)} background:var(--surface); color:var(--ink); padding:24px 28px 32px;
           font-family:Inter,"Helvetica Neue",Arial,sans-serif; border-radius:12px; }}
  .bench:has(#bench-dark:checked) {{ {css_vars(dark)} }}
  .bench header {{ display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; }}
  .bench h2 {{ margin:0; font-size:20px; font-weight:700; }}
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
  .bench td:nth-child(4) {{ font-weight:600; }}
  .bench .note {{ color:var(--ink-3); font-size:12px; margin-top:14px; }}
  .bench h3 {{ margin:32px 0 0; font-size:16px; font-weight:650; }}
</style>
<div class="bench">
  <header>
    <h2>pandera TensorDict validation overhead</h2>
    <div class="toggle" role="radiogroup" aria-label="Color theme">
      <input type="radio" name="bench-theme" id="bench-light" checked><label for="bench-light">Light</label>
      <input type="radio" name="bench-theme" id="bench-dark"><label for="bench-dark">Dark</label>
    </div>
  </header>
  <img class="chart-light" alt="Training time and validation overhead, light theme" src="data:image/png;base64,{light_b64}">
  <img class="chart-dark" alt="Training time and validation overhead, dark theme" src="data:image/png;base64,{dark_b64}">
  <table>
    <tr><th>steps</th><th>without (s)</th><th>with (s)</th><th>overhead</th><th>range (all runs)</th>
        <th>validate ms/batch</th><th>train ms/batch</th></tr>
    {rows_html}
  </table>
  <p class="note">Medians of {data["config"]["repeats"]} runs per length. Both modes train on the same batches;
  the baseline skips corrupt ones with a precomputed mask.</p>
  {sweep_html}
</div>
<script>
  if (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches)
    document.getElementById("bench-dark").checked = true;
</script>
"""


# {{docs-fragment report-and-main}}
@env.task(report=True)
async def plot(results: str, sweep: str) -> None:
    """Render both charts in both themes into the run's report tab."""
    data, sweep_data = json.loads(results), json.loads(sweep)
    pngs, step_pngs = {}, {}
    for theme in ("light", "dark"):
        out = Path(f"overhead-{theme}.png")
        render_plot(data, out, theme=theme)
        pngs[theme] = out.read_bytes()
        out = Path(f"step-size-{theme}.png")
        render_step_plot(sweep_data, out, theme=theme)
        step_pngs[theme] = out.read_bytes()
    await flyte.report.replace.aio(report_html(data, pngs["light"], pngs["dark"], sweep_data, step_pngs))
    await flyte.report.flush.aio()


@env.task
async def main(
    lengths: list[int] = [100, 300, 1_000, 3_000, 10_000],
    repeats: int = 5,
    batch_size: int = 256,
    obs_dim: int = 64,
    hidden: int = 256,
    corrupt_rate: float = 0.05,
    pool_size: int = 200,
    widths: list[int] = [64, 256, 1_024, 2_048, 4_096],
    sweep_repeats: int = 5,
    sweep_target_seconds: float = 15.0,
) -> tuple[str, str]:
    results = await run_benchmark(lengths, repeats, batch_size, obs_dim, hidden, corrupt_rate, pool_size)
    sweep = await run_step_size_sweep(
        widths, sweep_repeats, batch_size, obs_dim, corrupt_rate, pool_size, sweep_target_seconds
    )
    await plot(results, sweep)
    return results, sweep
# {{/docs-fragment report-and-main}}


def write_outputs(outputs, out_dir: Path) -> None:
    """Write main's two JSON outputs and render every chart locally, in both themes."""
    results, sweep = outputs
    (out_dir / "results.json").write_text(results)
    (out_dir / "step-size.json").write_text(sweep)
    data, sweep_data = json.loads(results), json.loads(sweep)
    render_plot(data, out_dir / "overhead.png")
    render_plot(data, out_dir / "overhead-dark.png", theme="dark")
    render_step_plot(sweep_data, out_dir / "step-size.png")
    render_step_plot(sweep_data, out_dir / "step-size-dark.png", theme="dark")
    print(f"Wrote results.json, step-size.json, and the charts to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run in-process instead of on the cluster")
    parser.add_argument("--lengths", type=int, nargs="+", default=[100, 300, 1_000, 3_000, 10_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--obs-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--corrupt-rate", type=float, default=0.05)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 256, 1_024, 2_048, 4_096])
    parser.add_argument("--sweep-repeats", type=int, default=5)
    parser.add_argument("--sweep-target-seconds", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    kwargs = dict(
        lengths=args.lengths, repeats=args.repeats, batch_size=args.batch_size,
        obs_dim=args.obs_dim, hidden=args.hidden, corrupt_rate=args.corrupt_rate, widths=args.widths,
        sweep_repeats=args.sweep_repeats, sweep_target_seconds=args.sweep_target_seconds,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.local:
        flyte.init()
        run = flyte.with_runcontext(mode="local").run(main, **kwargs)
        outputs = run.outputs()
    else:
        flyte.init_from_config()
        run = flyte.run(main, **kwargs)
        print(f"Run: {run.url}")
        run.wait()
        outputs = run.outputs()

    write_outputs(outputs, args.out)
