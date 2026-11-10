"""Reports and the artifact card for the G1 running policy: curves, filmstrips, clips, and tiles."""

import time

import viz
from g1 import video


def _curves(history: list[dict], width: int = 420) -> str:
    xs = [h["step"] for h in history]
    return viz.row(
        viz.panel("Eval reward", viz.line_chart(
            [viz.Series("eval reward", [h["reward"] for h in history], 1, xs=xs)],
            width=width, height=200)),
        viz.panel("Velocity tracking reward", viz.line_chart(
            [viz.Series("tracking_lin_vel", [h.get("tracking_lin_vel") for h in history], 3, xs=xs)],
            width=width, height=200),
            "The share of the reward for matching the commanded speed. Commands run 0 to max speed."),
    )


def _filmstrip(snaps: list[dict]) -> str:
    if not snaps:
        return "<p class='note'>no snapshots yet</p>"
    cells = "".join(
        f"<figure style='margin:0'><img src='data:image/jpeg;base64,{s['still']}' alt='step {s['step']:,}' "
        f"style='width:100%;border-radius:6px'/><figcaption class='note'>{viz.compact(s['step'])} steps · "
        f"{s['speed']:.1f} m/s</figcaption></figure>"
        for s in snaps
    )
    return f"<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px'>{cells}</div>"


def _clip_panel(title: str, clip: dict, max_steps: int) -> str:
    fell = viz.status("critical", f"fell at step {clip['steps']}") if clip["fell"] else viz.status("good", f"upright {max_steps}/{max_steps}")
    return viz.panel(title, clip["html"] + viz.table(
        ["commanded", "reached", "flight", "distance"],
        [(f"{clip['command'][0]:.1f} m/s", f"{clip['speed']:.2f} m/s", f"{clip['flight']:.0%}", f"{clip['distance']:.1f} m")])
        + f"<p class='note'>{fell}</p>")


def final_html(ckpt: dict, clips: dict[str, dict], run_name: str) -> str:
    history, run, walk, rand = ckpt["history"], clips["run"], clips["walk"], clips["random"]
    sps = ckpt["num_timesteps"] / max(ckpt["train_s"], 1)
    return viz.page(
        "Unitree G1 learns to run",
        f"Brax PPO over MJX: {ckpt['num_envs']:,} G1s simulated in parallel on one GPU, "
        f"{viz.compact(ckpt['num_timesteps'])} env steps in {ckpt['train_s'] / 60:.0f} min. "
        "Commands run from standing to the max speed; the replays hold one command each.",
        viz.tiles([
            ("Running speed", f"{run['speed']:.2f} m/s", f"commanded {run['command'][0]:.1f} m/s"),
            ("Flight phase", f"{run['flight']:.0%}", f"of steps with both feet off the ground (walk {walk['flight']:.0%})"),
            ("Throughput", f"{viz.compact(sps)}/s", "env steps per second of training"),
            ("Best eval reward", f"{max(h['reward'] for h in history):.1f}", f"first eval {history[0]['reward']:.1f}"),
        ], hero=True),
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px'>"
        + _clip_panel("Random policy", rand, ckpt["replay_steps"])
        + _clip_panel(f"Trained, walk command ({walk['command'][0]:.1f} m/s)", walk, ckpt["replay_steps"])
        + _clip_panel(f"Trained, run command ({run['command'][0]:.1f} m/s)", run, ckpt["replay_steps"])
        + "</div>",
        _curves(history),
        viz.panel("The gait over training", _filmstrip(ckpt["snapshots"]),
                  "One frame per mid-training snapshot, each filmed on the run command; the caption is the speed it reached."),
        viz.panel("Load it", viz.code(
            'from flyte.remote import Artifact\n'
            'policy = Artifact.get("g1-run-policy")   # pickle: Brax params + network config + history\n'
            f'# trained in run {run_name}')),
    )



def live_html(history: list[dict], snaps: list[dict], latest: dict, started: float,
              num_envs: int, num_timesteps: int, max_speed: float) -> str:
    """The training report: progress tiles, the latest filmed snapshot, and the reward curves."""
    clip = video(latest["mp4"], "latest snapshot") if latest else "<p class='note'>first snapshot after the first eval</p>"
    last = history[-1] if history else {"step": 0, "reward": 0.0}
    sps = last["step"] / max(time.time() - started, 1)
    return viz.page(
        "Training: G1 learns to run",
        f"{num_envs:,} parallel G1s on one GPU · commands 0 to {max_speed:.1f} m/s · "
        f"{viz.compact(num_timesteps)} steps planned",
        viz.tiles([
            ("Env steps", viz.compact(last["step"]), f"of {viz.compact(num_timesteps)}"),
            ("Eval reward", f"{last['reward']:.1f}"),
            ("Throughput", f"{viz.compact(sps)}/s", "env steps per second"),
            ("Elapsed", f"{(time.time() - started) / 60:.0f} min"),
        ], hero=True),
        viz.row(
            viz.panel(f"Latest snapshot (run command, {max_speed:.1f} m/s)", clip,
                      f"{viz.compact(latest['step'])} steps · reached {latest['speed']:.2f} m/s · "
                      f"flight {latest['flight']:.0%}" if latest else ""),
            viz.panel("Gait over training", _filmstrip(snaps)),
        ),
        _curves(history),
    )


def policy_card(ckpt: dict, clips: dict[str, dict], run_name: str) -> str:
    run, walk = clips["run"], clips["walk"]
    xs = [h["step"] for h in ckpt["history"]]
    stills = "".join(
        f"<img src='data:image/jpeg;base64,{c['still']}' alt='{k}' style='width:32%;border-radius:6px;margin:0 1% 6px 0'/>"
        for k, c in clips.items())
    return viz.card(
        "Unitree G1 running policy (Brax PPO)",
        f"Trained in run <code>{viz.esc(run_name)}</code> on {ckpt['num_envs']:,} parallel MJX environments, "
        f"{viz.compact(ckpt['num_timesteps'])} steps, commands 0 to {ckpt['max_speed']:.1f} m/s.",
        ["JAX", "Brax PPO", "MuJoCo MJX", "Unitree G1", "locomotion", ".pkl"],
        viz.tiles([
            ("Running speed", f"{run['speed']:.2f} m/s", f"commanded {run['command'][0]:.1f}"),
            ("Walking speed", f"{walk['speed']:.2f} m/s", f"commanded {walk['command'][0]:.1f}"),
            ("Flight phase", f"{run['flight']:.0%}", "at the run command"),
            ("Training time", f"{ckpt['train_s'] / 60:.0f} min", "one GPU"),
        ], hero=True),
        viz.panel("Random, walk, run", stills),
        viz.panel("Eval reward", viz.line_chart(
            [viz.Series("eval reward", [h["reward"] for h in ckpt["history"]], 1, xs=xs)], )),
        viz.panel("Load it", viz.code(
            'from flyte.remote import Artifact\n'
            'policy = Artifact.get("g1-run-policy")\n'
            '# pickle.load(...) -> {"params", "network_factory", "history", ...}')),
    )
