# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.10.1,<2.11",
#    "gymnasium==1.*",
#    "matplotlib",
#    "pandera[torch]==0.34.0",
#    "torch",
# ]
# main = "main"
# params = "rates=[0.05] seeds=3"
# ///
"""What does it cost to NOT validate the data going into a model?

The companion overhead benchmark (../pandera_validation_overhead) measures
what validation costs. This one measures the other side: what happens to
training when corrupt batches reach the model, compared with a loop that
validates every batch with pandera and skips the ones that fail.

Task: CartPole-v1 (gymnasium), REINFORCE with a small MLP policy. Each update
collects a batch of episodes into a TensorDict. A fraction of batches is
corrupted on the way to the update, in one of two ways:

- nan_reward: one reward in the batch is NaN. It reaches the loss, the weights
  go NaN, and the next rollout crashes. The loop rolls back to its last good
  checkpoint and keeps going, so the cost is the work thrown away.
- obs_scale: the whole batch's observations are 100x too large (a units or
  normalization bug). Nothing crashes; the update just pushes the policy the
  wrong way, so the cost is slower or failed learning.

Two modes per cell, same seed and same corruption schedule:

- without pandera: corrupt batches go straight into the update.
- with pandera: `Transition.validate(td, inplace=True)` on every batch, and the
  batch is skipped on `SchemaError`.

The cost is measured in environment steps to reach a 475 return (CartPole-v1's
"solved" threshold, averaged over the last 20 episodes), counting every step
collected, including those thrown away by rollbacks or skipped batches. Steps
are deterministic for a given seed, so the comparison doesn't depend on how
busy the node is. Wall-clock time and time spent validating are recorded too.

Run on the cluster configured in ~/.flyte/config.yaml:

    uv run main.py

Run locally:

    uv run main.py --local --seeds 3 --rates 0.05
"""

import argparse
import asyncio
import base64
import json
import statistics
import time
from pathlib import Path

import flyte

# {{docs-fragment image-and-env}}
image = (
    flyte.Image.from_debian_base(python_version=(3, 12), name="pandera-cost-of-not-validating")
    .with_pip_packages("torch==2.*", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("pandera[torch]==0.34.0", "gymnasium==1.*", "matplotlib")
    .with_apt_packages("fonts-inter")
)

# Each run is single-threaded and small, so cells run in parallel as separate
# tasks rather than as threads in one pod.
env = flyte.TaskEnvironment(
    name="pandera_cost_of_not_validating",
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory="2Gi"),
)

OBS_BOUND = 10.0  # CartPole terminates long before any observation gets near this
TARGET_RETURN = 475.0
RETURN_WINDOW = 20
EPISODES_PER_UPDATE = 8
CKPT_EVERY = 10  # updates between checkpoints
MAX_ENV_STEPS = 1_000_000  # a run that hasn't reached the target by here counts as failed
CORRUPTIONS = ("nan_reward", "obs_scale")
# {{/docs-fragment image-and-env}}


# {{docs-fragment schema}}
def build_schema():
    import torch
    import pandera.tensordict as pa

    class Transition(pa.TensorDictModel):
        observation: torch.float32 = pa.Field(shape=(None, 4), ge=-OBS_BOUND, le=OBS_BOUND)
        action: torch.int64 = pa.Field(shape=(None,), isin=[0, 1])
        reward: torch.float32 = pa.Field(shape=(None,), ge=0.0, le=1.0)
        episode: torch.int64 = pa.Field(shape=(None,))

        class Config:
            batch_size = (None,)

    return Transition
# {{/docs-fragment schema}}


def train(corruption: str | None, rate: float, validate: bool, seed: int) -> dict:
    """Train one REINFORCE policy until it reaches the target return or runs out of steps."""
    import copy
    import random

    import gymnasium as gym
    import numpy as np
    import torch
    from tensordict import TensorDict
    import pandera.tensordict as pa

    schema = build_schema() if validate else None
    torch.manual_seed(seed)
    envs = [gym.make("CartPole-v1") for _ in range(EPISODES_PER_UPDATE)]
    policy = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    # A separate stream for corruption, so both modes see the same schedule.
    corrupt_rng = random.Random(10_000 + seed)

    checkpoints = [(copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict()), 0)]
    returns: list[float] = []
    env_steps = wasted_steps = skipped_steps = 0
    crashes = skipped = corrupted = updates = poisoned_checkpoints = 0
    restored_at = 0  # env_steps at the last rollback, so repeated rollbacks don't double-count
    episode_seed = seed * 1_000_000
    reached_at = None
    validate_s = 0.0
    start = time.perf_counter()

    while env_steps < MAX_ENV_STEPS:
        try:
            # Roll out the batch's episodes in lockstep: one policy call per tick.
            current, buf = {}, {}
            for e in range(EPISODES_PER_UPDATE):
                obs, _ = envs[e].reset(seed=episode_seed)
                episode_seed += 1
                current[e] = (obs, 0.0)
                buf[e] = ([], [], [])
            while current:
                ids = list(current)
                batch_obs = torch.as_tensor(np.array([current[e][0] for e in ids]), dtype=torch.float32)
                with torch.no_grad():
                    probs = torch.softmax(policy(batch_obs), dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(1).tolist()  # raises on NaN weights
                for e, a in zip(ids, actions):
                    obs, ret = current[e]
                    next_obs, r, terminated, truncated, _ = envs[e].step(a)
                    buf[e][0].append(obs)
                    buf[e][1].append(a)
                    buf[e][2].append(r)
                    if terminated or truncated:
                        returns.append(ret + r)
                        del current[e]
                    else:
                        current[e] = (next_obs, ret + r)

            obs_l, act_l, rew_l, ep_l = [], [], [], []
            for e in range(EPISODES_PER_UPDATE):
                obs_l += buf[e][0]
                act_l += buf[e][1]
                rew_l += buf[e][2]
                ep_l += [e] * len(buf[e][2])
            n = len(rew_l)
            env_steps += n
            td = TensorDict(
                {
                    "observation": torch.as_tensor(np.array(obs_l), dtype=torch.float32),
                    "action": torch.as_tensor(act_l, dtype=torch.int64),
                    "reward": torch.as_tensor(rew_l, dtype=torch.float32),
                    "episode": torch.as_tensor(ep_l, dtype=torch.int64),
                },
                batch_size=[n],
            )

            # {{docs-fragment corrupt-and-validate}}
            if corruption is not None and corrupt_rng.random() < rate:
                corrupted += 1
                if corruption == "nan_reward":
                    td["reward"][corrupt_rng.randrange(n)] = float("nan")
                elif corruption == "obs_scale":
                    td["observation"] = td["observation"] * 100.0

            if schema is not None:
                t0 = time.perf_counter()
                try:
                    schema.validate(td, inplace=True)
                    is_valid = True
                except pa.SchemaError:
                    is_valid = False
                validate_s += time.perf_counter() - t0
                if not is_valid:
                    skipped += 1
                    skipped_steps += n
                    continue
            # {{/docs-fragment corrupt-and-validate}}

            # Discounted, normalized returns per episode.
            r = td["reward"].numpy()
            ep = td["episode"].numpy()
            g_arr = np.zeros(n, dtype=np.float32)
            g = 0.0
            for i in range(n - 1, -1, -1):
                if i == n - 1 or ep[i] != ep[i + 1]:
                    g = 0.0
                g = r[i] + 0.99 * g
                g_arr[i] = g
            g_t = torch.from_numpy(g_arr)
            g_t = (g_t - g_t.mean()) / (g_t.std() + 1e-8)

            log_prob = torch.log_softmax(policy(td["observation"]), dim=-1)
            log_prob = log_prob.gather(1, td["action"].unsqueeze(1)).squeeze(1)
            loss = -(log_prob * g_t).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            if updates % CKPT_EVERY == 0:
                checkpoints.append((copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict()), env_steps))

            if len(returns) >= RETURN_WINDOW and statistics.mean(returns[-RETURN_WINDOW:]) >= TARGET_RETURN:
                reached_at = env_steps
                break
        # {{docs-fragment rollback}}
        except (ValueError, RuntimeError):
            # NaN weights: the rollout raised. Roll back to the newest checkpoint with
            # finite weights. A checkpoint saved right after a NaN update is poisoned,
            # and restoring it would crash again forever, so step back past it the way
            # an operator would.
            crashes += 1
            while not _finite(checkpoints[-1]):
                checkpoints.pop()
                poisoned_checkpoints += 1
            model_state, opt_state, mark = checkpoints[-1]
            wasted_steps += env_steps - max(mark, restored_at)
            restored_at = env_steps
            # Load copies. Optimizer.load_state_dict keeps references to the tensors it
            # is given, so loading the stored state directly lets the next NaN update
            # write into the checkpoint itself and poison it for every later rollback.
            policy.load_state_dict(copy.deepcopy(model_state))
            optimizer.load_state_dict(copy.deepcopy(opt_state))
        # {{/docs-fragment rollback}}

    return {
        "corruption": corruption,
        "rate": rate,
        "validate": validate,
        "seed": seed,
        "reached": reached_at is not None,
        "env_steps": env_steps,
        "wasted_steps": wasted_steps,
        "skipped_steps": skipped_steps,
        "crashes": crashes,
        "poisoned_checkpoints": poisoned_checkpoints,
        "corrupted_batches": corrupted,
        "skipped_batches": skipped,
        "updates": updates,
        "final_return": statistics.mean(returns[-RETURN_WINDOW:]) if returns else 0.0,
        "seconds": time.perf_counter() - start,
        "validate_seconds": validate_s,
    }


def _finite(checkpoint) -> bool:
    """True if a checkpoint's weights and optimizer state hold no NaN or inf."""
    import torch

    model_state, opt_state, _ = checkpoint
    tensors = list(model_state.values()) + [
        v for state in opt_state["state"].values() for v in state.values() if torch.is_tensor(v)
    ]
    return all(torch.isfinite(t).all() for t in tensors)


# {{docs-fragment run-cell}}
@env.task
async def run_cell(corruption: str | None, rate: float, seeds: int) -> str:
    """Every seed of one (corruption, rate) cell, in both modes."""
    import torch

    torch.set_num_threads(1)
    runs = [train(corruption, rate, validate, seed) for seed in range(seeds) for validate in (False, True)]
    for row in runs:
        print(
            f"{corruption}@{rate:.0%} seed={row['seed']} validate={row['validate']} reached={row['reached']} "
            f"steps={row['env_steps']:,} crashes={row['crashes']} skipped={row['skipped_batches']}",
            flush=True,
        )
    return json.dumps(runs)
# {{/docs-fragment run-cell}}


def summarize(runs: list[dict]) -> list[dict]:
    """One row per (corruption, rate, mode): medians and quartiles over seeds.

    Runs that never reached the target count at the step budget, so the median is
    a lower bound whenever a cell has failures; `reached` says how many made it.
    """
    cells: dict[tuple, list[dict]] = {}
    for r in runs:
        cells.setdefault((r["corruption"], r["rate"], r["validate"]), []).append(r)
    rows = []
    for (corruption, rate, validate), rs in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        steps = sorted(r["env_steps"] for r in rs)
        q = statistics.quantiles(steps, n=4) if len(steps) > 1 else [steps[0]] * 3
        rows.append({
            "corruption": corruption,
            "rate": rate,
            "validate": validate,
            "seeds": len(rs),
            "reached": sum(r["reached"] for r in rs),
            "steps_median": statistics.median(steps),
            "steps_q1": q[0],
            "steps_q3": q[2],
            "wasted_median": statistics.median(r["wasted_steps"] for r in rs),
            "crashes_median": statistics.median(r["crashes"] for r in rs),
            "final_return_median": statistics.median(r["final_return"] for r in rs),
            "validate_share": sum(r["validate_seconds"] for r in rs) / max(sum(r["seconds"] for r in rs), 1e-9),
        })
    return rows


# {{docs-fragment main}}
@env.task
async def main(rates: list[float] = [0.01, 0.05, 0.2], seeds: int = 30) -> str:
    import os
    import platform

    cells = [(None, 0.0)] + [(c, r) for c in CORRUPTIONS for r in rates]
    outs = await asyncio.gather(*(run_cell(c, r, seeds) for c, r in cells))
    runs = [row for out in outs for row in json.loads(out)]

    import gymnasium
    import pandera
    import tensordict
    import torch

    result = {
        "config": {
            "rates": rates, "seeds": seeds, "target_return": TARGET_RETURN, "return_window": RETURN_WINDOW,
            "episodes_per_update": EPISODES_PER_UPDATE, "ckpt_every": CKPT_EVERY, "max_env_steps": MAX_ENV_STEPS,
            "obs_bound": OBS_BOUND,
        },
        "environment": {
            "torch": torch.__version__, "pandera": pandera.__version__, "tensordict": tensordict.__version__,
            "gymnasium": gymnasium.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    out = json.dumps(result, indent=2)
    await plot(out)
    return out
# {{/docs-fragment main}}


# Categorical slots 1 and 2 of the dataviz reference palette, the same pair the
# overhead benchmark uses: blue = without pandera, orange = with pandera.
THEMES = {
    "light": {
        "without": "#2a78d6", "with": "#eb6834", "surface": "#fcfcfb", "ink": "#0b0b0b",
        "ink_secondary": "#52514e", "ink_muted": "#8a897f", "grid": "#e6e5e0",
        "without_band": "#c5dbf4", "with_band": "#f8d3c2",
    },
    "dark": {
        "without": "#3987e5", "with": "#d95926", "surface": "#1a1a19", "ink": "#ffffff",
        "ink_secondary": "#c3c2b7", "ink_muted": "#8f8e86", "grid": "#333331",
        "without_band": "#1d3550", "with_band": "#4a2a1c",
    },
}

PANEL_TITLES = {
    "nan_reward": "A NaN reward: the run crashes and rolls back",
    "obs_scale": "Observations 100x too large: nothing crashes",
}


def render_plot(result: dict, out: Path, theme: str = "light") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    c = THEMES[theme]
    cfg, env_info = result["config"], result["environment"]
    rows = result["summary"]
    clean = {r["validate"]: r for r in rows if r["corruption"] is None}

    plt.rcParams.update({
        "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.edgecolor": c["grid"], "axes.labelcolor": c["ink_secondary"],
        "axes.titlesize": 12.5, "axes.titleweight": "semibold", "axes.titlecolor": c["ink"], "axes.titlepad": 12,
        "xtick.color": c["ink_secondary"], "ytick.color": c["ink_secondary"],
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor=c["surface"], sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.2, wspace=0.12)

    rates = [0.0] + list(cfg["rates"])
    x = list(range(len(rates)))
    budget = cfg["max_env_steps"]
    # Compute relative to a clean run, so the panels read as "how many times the
    # compute", and a log axis keeps a 1.1x gap and a 10x gap both visible.
    base = clean[False]["steps_median"]
    rel = lambda v: v / base
    worst, worst_censored = 0.0, False
    for ax, corruption in zip(axes, CORRUPTIONS):
        ax.set_facecolor(c["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for validate, color, band in ((False, c["without"], c["without_band"]), (True, c["with"], c["with_band"])):
            cell = [clean[validate]] + sorted(
                (r for r in rows if r["corruption"] == corruption and r["validate"] == validate), key=lambda r: r["rate"]
            )
            med = [rel(r["steps_median"]) for r in cell]
            ax.fill_between(x, [rel(r["steps_q1"]) for r in cell], [rel(r["steps_q3"]) for r in cell], color=band,
                            linewidth=0, alpha=0.9, zorder=1)
            ax.plot(x, med, color=color, linewidth=2, zorder=3)
            ax.scatter(x, med, s=70, color=color, edgecolor=c["surface"], linewidth=2, zorder=4)
            if not validate:
                for r, m in zip(cell, med):
                    if m > worst:
                        worst, worst_censored = m, r["reached"] < r["seeds"]
            # Label the last point, and say how many runs never got there (their
            # steps count at the budget, so the point is a lower bound).
            for xi, r, m in zip(x, cell, med):
                failed = r["seeds"] - r["reached"]
                label = f"{'≥' if failed else ''}{m:.1f}x"
                if failed:
                    label += f"\n{failed}/{r['seeds']} runs never solved it"
                if xi == x[-1] or failed:
                    ax.annotate(label, (xi, m), xytext=(-10, 0), textcoords="offset points", ha="right",
                                va="center", fontsize=9.5, color=c["ink"],
                                bbox={"boxstyle": "round,pad=0.2", "facecolor": c["surface"], "edgecolor": "none"})
        ax.set_xticks(x, [f"{r:.0%}" for r in rates])
        ax.set_xlabel("share of batches corrupted")
        ax.set_title(PANEL_TITLES[corruption], loc="left")

    for ax in axes:
        ax.axhline(rel(budget), color=c["ink_muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    axes[0].annotate(f"{budget / 1e6:g}M-step budget", (x[0], rel(budget)), xytext=(0, 5),
                     textcoords="offset points", fontsize=9, color=c["ink_muted"], va="bottom")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.7, rel(budget) * 1.6)
    ticks = [t for t in (1, 2, 5, 10) if t < rel(budget) * 1.6]
    axes[0].yaxis.set_major_locator(FixedLocator(ticks))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    axes[0].set_ylabel(f"compute to solve, relative to a clean run ({base / 1000:.0f}K steps)")
    axes[0].legend(
        handles=[
            Line2D([], [], color=c["without"], linewidth=2, marker="o", markersize=8,
                   markeredgecolor=c["surface"], label="without pandera"),
            Line2D([], [], color=c["with"], linewidth=2, marker="o", markersize=8,
                   markeredgecolor=c["surface"], label="with pandera (validate, skip bad batches)"),
        ],
        frameon=False, loc="upper left", bbox_to_anchor=(0, 0.8), labelcolor=c["ink_secondary"], fontsize=10,
    )

    vshare = statistics.median(r["validate_share"] for r in rows if r["validate"])
    fig.text(0.08, 0.94, f"Training on corrupt batches took {'more than ' if worst_censored else 'up to '}{int(worst)}x the compute",
             fontsize=18, fontweight="bold", color=c["ink"])
    fig.text(0.08, 0.895,
             f"Environment steps to solve CartPole with REINFORCE when a share of batches is corrupt, counting steps "
             f"lost to rollbacks and\nskipped batches. Validating every batch took {vshare:.1%} of wall-clock time. "
             "Lines are the median over seeds; bands span the middle half.",
             fontsize=11.5, color=c["ink_secondary"], va="top", linespacing=1.45)
    fig.text(
        0.08, 0.025,
        f"{cfg['seeds']} seeds per point. Batch of {cfg['episodes_per_update']} episodes per update, checkpoint every "
        f"{cfg['ckpt_every']} updates, rollback to the newest finite checkpoint on a crash. Runs that hadn't reached the "
        f"target by {budget / 1e6:g}M steps\ncount at that budget. pandera {env_info['pandera']}, torch "
        f"{env_info['torch']}, gymnasium {env_info['gymnasium']}, {env_info['machine']}.",
        color=c["ink_muted"], fontsize=9, linespacing=1.5, va="bottom",
    )
    fig.savefig(out, dpi=200, facecolor=c["surface"])
    plt.close(fig)


@env.task(report=True)
async def plot(result: str) -> None:
    """Render the chart in both themes into the run's report tab."""
    data = json.loads(result)
    imgs = {}
    for theme in ("light", "dark"):
        out = Path(f"cost-{theme}.png")
        render_plot(data, out, theme=theme)
        imgs[theme] = base64.b64encode(out.read_bytes()).decode()
    await flyte.report.replace.aio(
        f'<img style="max-width:100%" src="data:image/png;base64,{imgs["light"]}">'
        f'<img style="max-width:100%" src="data:image/png;base64,{imgs["dark"]}">'
    )
    await flyte.report.flush.aio()


def write_outputs(result: str, out_dir: Path) -> None:
    (out_dir / "results.json").write_text(result)
    data = json.loads(result)
    render_plot(data, out_dir / "cost.png")
    render_plot(data, out_dir / "cost-dark.png", theme="dark")
    print(f"Wrote results.json, cost.png, and cost-dark.png to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="run in-process instead of on the cluster")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.01, 0.05, 0.2])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.local:
        flyte.init()
        run = flyte.with_runcontext(mode="local").run(main, rates=args.rates, seeds=args.seeds)
    else:
        flyte.init_from_config()
        run = flyte.run(main, rates=args.rates, seeds=args.seeds)
        print(f"Run: {run.url}")
        run.wait()
    outputs = run.outputs()
    write_outputs(outputs if isinstance(outputs, str) else outputs[0], args.out)
