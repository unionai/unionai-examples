# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
# ]
# main = "learn_to_run"
# params = "num_timesteps=20000000 publish=False"
# ///
"""Teach a Unitree G1 humanoid to run with Brax PPO over MJX, thousands of robots on one GPU.

    train_policy (GPU) -> 3 x render_replay (CPU, in parallel) -> publish_policy

MJX compiles the physics into the same XLA program as the learner, so 8,192 simulated robots
step in parallel on one GPU and the policy gets the hundreds of millions of steps a humanoid
gait needs. The training report streams the reward curve and films the current policy every
few evals. When training ends, three CPU replays film a random policy and the trained policy
at walking and running speed, and the policy is published as an artifact with a model card.

    python learn_to_run.py                       # 150M steps, about 20 minutes on an L40S
    python learn_to_run.py --smoke               # 20M steps, about 8 minutes, falls over
    python learn_to_run.py --checkpoint <uri>    # replay and report an existing policy
"""

import asyncio
import json
import logging
import pickle
import sys
import time

import flyte
import flyte.artifacts as artifacts
import flyte.report
import viz
from flyte.io import File
from g1 import (
    BASE_ENV,
    WALK_SPEED,
    encode_mp4,
    inference_fn,
    make_env,
    ppo_config,
    render_side,
    rollout,
    still,
    video,
)
from reports import final_html, live_html, policy_card

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# {{docs-fragment image}}
# brax 0.14.2 still calls jax.device_put_replicated, which jax removed in 0.10, so jax stays
# on 0.9.x until brax moves off the old pmap API.
image = (
    flyte.Image.from_debian_base(name="g1-mjx")
    .with_apt_packages("git", "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libgles2", "libglx-mesa0")
    .with_pip_packages(
        "jax[cuda12]==0.9.2", "mujoco==3.17.0", "mujoco-mjx==3.17.0", "brax==0.14.2", "playground==0.2.0",
        "numpy", "Pillow", "av",
    )
    # Bake the robot models into the image. Otherwise every pod clones mujoco_menagerie at startup.
    .with_commands(["python -c 'from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()'"])
)
# {{/docs-fragment image}}

# {{docs-fragment envs}}
EGL = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

# Physics and gradients, both on the GPU. MJX's Warp kernels allocate outside JAX's memory
# pool, so JAX must not preallocate most of the card.
gpu_env = flyte.TaskEnvironment(
    name="g1-run-train",
    image=image,
    resources=flyte.Resources(cpu=8, memory="48Gi", gpu="L40s:1", disk="50Gi"),
    env_vars={"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75", **EGL},
)

# Replays. MJX has no renderer, so the policy runs on MJX's JAX backend on CPU and MuJoCo's
# renderer draws it over headless EGL.
render_env = flyte.TaskEnvironment(
    name="g1-run-render",
    image=image,
    resources=flyte.Resources(cpu=4, memory="16Gi"),
    env_vars={"JAX_PLATFORMS": "cpu", **EGL},
)

# The driver holds its pod while the other tasks run, so it stays small and CPU-only.
driver_env = flyte.TaskEnvironment(
    name="g1-run-driver",
    image=image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    env_vars={"JAX_PLATFORMS": "cpu"},
    depends_on=[gpu_env, render_env],
)
# {{/docs-fragment envs}}


# {{docs-fragment train}}
@gpu_env.task(report=True, retries=1)
async def train_policy(
    num_timesteps: int = 150_000_000,
    num_envs: int = 8192,
    num_evals: int = 20,
    max_speed: float = 2.5,
    seed: int = 0,
    snapshot_every: int = 4,
    snapshot_steps: int = 200,
) -> File:
    """Train one G1 running policy on one GPU and return its checkpoint.

    The reward curve streams into this task's report at every eval. Every ``snapshot_every``
    evals, the current policy is filmed on the run command.
    """
    import functools

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper

    log.info(f"devices: {jax.devices()}")
    env = make_env(max_speed)
    params = ppo_config(num_timesteps, num_envs, num_evals)
    net_cfg = dict(params.network_factory)
    history: list[dict] = []
    snaps: list[dict] = []
    latest: dict = {}
    started = time.time()
    evals = {"n": 0}

    def progress(step: int, metrics: dict) -> None:
        row = {"step": int(step), "reward": float(metrics.get("eval/episode_reward", 0.0)),
               "elapsed_s": round(time.time() - started, 1)}
        for k in ("tracking_lin_vel", "feet_air_time"):
            if f"eval/episode_reward/{k}" in metrics:
                row[k] = float(metrics[f"eval/episode_reward/{k}"])
        history.append(row)
        log.info(f"  step {row['step']:>12,} | reward {row['reward']:8.2f} | {row['step'] / max(row['elapsed_s'], 1):,.0f} steps/s")
        flyte.report.replace(live_html(history, snaps, latest, started, num_envs, num_timesteps, max_speed))
        flyte.report.flush()

    def policy_params_fn(step: int, make_policy, p) -> None:
        # The only Brax hook that hands over live weights. Filming must never kill the run.
        evals["n"] += 1
        if not snapshot_every or step == 0 or (evals["n"] - 1) % snapshot_every:
            return
        try:
            policy = jax.jit(make_policy(p, deterministic=True))
            r = rollout(env, lambda obs, key: policy(obs, key)[0], (max_speed, 0.0, 0.0), snapshot_steps)
            frames = render_side(env, r["traj"], 240, 320)
            latest.update(step=step, speed=r["speed"], flight=r["flight"], mp4=encode_mp4(frames, int(round(1 / env.dt))))
            snaps.append({"step": step, "speed": r["speed"], "flight": r["flight"], "still": still(frames[len(frames) * 2 // 3])})
        except Exception as exc:
            log.warning(f"  snapshot at {step:,} failed ({exc}); training continues")

    _, trained, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **net_cfg),
        randomization_fn=registry.get_domain_randomizer(BASE_ENV),
        # Playground's wrapper applies the domain randomization. Brax's default drops it.
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=seed,
        **{k: v for k, v in params.items() if k != "network_factory"},
    )
    train_s = time.time() - started
    log.info(f"trained {num_timesteps:,} steps in {train_s / 60:.1f} min")

    path = "/tmp/g1_run_policy.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "params": jax.device_get(trained), "network_factory": net_cfg, "history": history,
            "snapshots": snaps, "max_speed": max_speed, "num_timesteps": num_timesteps,
            "num_envs": num_envs, "seed": seed, "train_s": round(train_s, 1),
        }, f)
    return await File.from_local(path)
# {{/docs-fragment train}}


# {{docs-fragment replay}}
@render_env.task(retries=2)
async def render_replay(checkpoint: File, speed: float, policy: str = "trained", steps: int = 500) -> str:
    """Roll one policy out on one forward command, and return the clip and what it measured."""
    import jax

    with open(await checkpoint.download(), "rb") as f:
        ckpt = pickle.load(f)
    env = make_env(ckpt["max_speed"], impl="jax")
    if policy == "trained":
        fn = inference_fn(ckpt["network_factory"], ckpt["params"], env)

        def act(obs, key):
            return fn(obs, key)[0]
    else:
        def act(obs, key):
            return jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)

    r = rollout(env, act, (speed, 0.0, 0.0), steps)
    frames = render_side(env, r.pop("traj"), 360, 480)
    label = f"{policy} policy, {speed:.1f} m/s command"
    log.info(f"{label}: {r['steps']}/{steps} steps, {r['speed']:.2f} m/s, flight {r['flight']:.0%}, {r['distance']:.1f} m")
    return json.dumps({**r, "policy": policy, "command": [speed, 0.0, 0.0],
                       "html": video(encode_mp4(frames, int(round(1 / env.dt))), label),
                       "still": still(frames[len(frames) * 2 // 3], 360)})
# {{/docs-fragment replay}}


# {{docs-fragment publish}}
@driver_env.task(produces_artifacts=True)
async def publish_policy(checkpoint: File, clips_json: str) -> File:
    """Register the checkpoint as the model artifact ``g1-run-policy``, with a model card."""
    clips = json.loads(clips_json)
    with open(await checkpoint.download(), "rb") as f:
        ckpt = pickle.load(f)
    run_name = flyte.ctx().action.run_name
    card = await artifacts.Card.create_from.aio(content=policy_card(ckpt, clips, run_name), format="html", card_type="model")
    metadata = artifacts.Metadata.create_model_metadata(
        name="g1-run-policy",
        description=f"Unitree G1 running policy: {clips['run']['speed']:.2f} m/s at a {clips['run']['command'][0]:.1f} m/s command",
        framework="JAX",
        model_type="actor-critic MLP",
        architecture="Brax PPO, asymmetric actor-critic",
        task="humanoid locomotion",
        modality=("proprioception",),
        serial_format="pkl",
        card=card,
        attrs={"run_speed": f"{clips['run']['speed']:.2f}", "flight": f"{clips['run']['flight']:.2f}",
               "env_steps": str(ckpt["num_timesteps"])},
    )
    return artifacts.new(checkpoint, metadata)
# {{/docs-fragment publish}}


# {{docs-fragment driver}}
@driver_env.task(report=True)
async def learn_to_run(
    num_timesteps: int = 150_000_000,
    num_envs: int = 8192,
    num_evals: int = 20,
    max_speed: float = 2.5,
    seed: int = 0,
    replay_steps: int = 500,
    publish: bool = True,
    checkpoint: File | None = None,
) -> str:
    """Train a G1 running policy, replay it next to a random policy, and publish it.

    Pass ``checkpoint`` (a ``train_policy`` output) to skip training and only replay and report.
    """
    if checkpoint is None:
        await flyte.report.replace.aio(viz.page(
            "Unitree G1 learns to run",
            f"Training {viz.compact(num_timesteps)} steps across {num_envs:,} parallel G1s on one GPU. "
            "The live curve and snapshots are in the train_policy action's report."), do_flush=True)
        checkpoint = await train_policy(num_timesteps=num_timesteps, num_envs=num_envs, num_evals=num_evals,
                                        max_speed=max_speed, seed=seed)
    with open(await checkpoint.download(), "rb") as f:
        ckpt = pickle.load(f)
    max_speed = ckpt["max_speed"]

    # Three independent CPU replays. A trained clip means little without the random one next to it.
    rand, walk, run = await asyncio.gather(
        render_replay(checkpoint, max_speed, "random", replay_steps),
        render_replay(checkpoint, WALK_SPEED, "trained", replay_steps),
        render_replay(checkpoint, max_speed, "trained", replay_steps),
    )
    clips = {"random": json.loads(rand), "walk": json.loads(walk), "run": json.loads(run)}
    ckpt["replay_steps"] = replay_steps
    await flyte.report.replace.aio(final_html(ckpt, clips, flyte.ctx().action.run_name), do_flush=True)

    if publish:
        await publish_policy(checkpoint, json.dumps({k: {kk: vv for kk, vv in c.items() if kk != "html"} for k, c in clips.items()}))

    summary = {k: {kk: c[kk] for kk in ("command", "speed", "flight", "distance", "steps", "fell")} for k, c in clips.items()}
    summary["best_reward"] = max(h["reward"] for h in ckpt["history"])
    summary["train_min"] = round(ckpt["train_s"] / 60, 1)
    return json.dumps(summary)
# {{/docs-fragment driver}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    if "--checkpoint" in sys.argv:
        uri = sys.argv[sys.argv.index("--checkpoint") + 1]
        run = flyte.run(learn_to_run, checkpoint=File.from_existing_remote(uri))
    elif "--smoke" in sys.argv:
        run = flyte.run(learn_to_run, num_timesteps=20_000_000, publish=False)
    else:
        run = flyte.run(learn_to_run)
    print(run.url)
# {{/docs-fragment run}}
