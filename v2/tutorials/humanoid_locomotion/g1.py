"""The G1 running environment, the PPO config, and the replay, filming, and encoding helpers.

The environment is DeepMind's tuned ``G1JoystickFlatTerrain`` from MuJoCo Playground. The robot
model comes from MuJoCo Menagerie (Unitree G1, BSD-3-Clause). Compared with upstream:

- forward commands run from 0 to ``max_speed`` (upstream: -1.0 to 1.0 m/s), so the robot never
  practises walking backwards;
- lateral commands narrow to +/-0.3 m/s (upstream +/-0.5) and yaw to +/-0.5 rad/s (upstream
  +/-1.0), so most of the training signal goes to running forward;
- the reward weights ``tracking_lin_vel`` at 1.5 (upstream 1.0) and ``feet_air_time`` at 2.5
  (upstream 2.0);
- the gait clock speeds up with the command, so a fast command asks for a faster cadence
  instead of ever-longer strides;
- ``njmax`` is 128 (upstream 90), enough constraint rows for a G1 in ground contact;
- MuJoCo-Warp's per-step solver-iteration warnings are off. Its other overflow warnings,
  including ``nefc > njmax``, stay on.
"""

import base64
import io

import viz

# {{docs-fragment env}}
BASE_ENV = "G1JoystickFlatTerrain"
WALK_SPEED = 1.0
# The gait clock: 1.25 Hz at a standstill (upstream's lower bound), rising with the command.
# At 2.5 m/s that is ~2.1 Hz, a 0.6 m step, which is a run for a 1.3 m robot.
CADENCE_HZ = (1.25, 2.25)
CADENCE_PER_MPS = 0.35


TERRAINS = {"flat": "flat_terrain", "rough": "rough_terrain"}


def env_config(max_speed: float, impl: str = "warp"):
    """Upstream's G1 config, with run-speed forward commands and the njmax fix."""
    from mujoco_playground import registry

    cfg = registry.get_default_config(BASE_ENV)
    cfg.impl = impl
    # Upstream ships njmax=90 and a G1 in ground contact needs ~93; on overflow MJX drops
    # constraints silently and the feet sink into the floor.
    cfg.njmax = 128
    cfg.lin_vel_x = [0.0, max_speed]   # forward only: walk through run
    cfg.lin_vel_y = [-0.3, 0.3]
    cfg.ang_vel_yaw = [-0.5, 0.5]
    # At run speed the tracking term is the one worth chasing, and air time pays for flight.
    cfg.reward_config.scales.tracking_lin_vel = 1.5
    cfg.reward_config.scales.feet_air_time = 2.5
    return cfg


def make_env(max_speed: float, impl: str = "warp", terrain: str = "flat"):
    import jax.numpy as jp
    from mujoco_playground._src.locomotion.g1 import joystick

    class G1Run(joystick.Joystick):
        """G1 joystick env whose gait clock speeds up with the commanded forward speed."""

        def __init__(self, **kw):
            super().__init__(**kw)
            if impl == "warp":
                from mujoco.mjx.third_party.mujoco_warp import OverflowType

                # MuJoCo-Warp printf's a solver-iteration warning from the kernel for every
                # world on every step: at 8,192 worlds that is most of the wall clock. Turn off
                # only the two iteration-limit warnings; the nefc > njmax one stays on.
                quiet = OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS
                self._mjx_model = self._mjx_model.tree_replace(
                    {"opt._impl.warn_overflow": int(OverflowType.ALL) & ~int(quiet)})

        def _phase_dt(self, command):
            hz = jp.clip(CADENCE_HZ[0] + CADENCE_PER_MPS * jp.abs(command[0]), *CADENCE_HZ)
            return (2 * jp.pi * self.dt * hz).reshape(1)

        def reset(self, rng):
            state = super().reset(rng)
            state.info["phase_dt"] = self._phase_dt(state.info["command"])
            return state

        def step(self, state, action):
            state = super().step(state, action)
            state.info["phase_dt"] = self._phase_dt(state.info["command"])
            return state

    return G1Run(task=TERRAINS[terrain], config=env_config(max_speed, impl))
# {{/docs-fragment env}}


def ppo_config(num_timesteps: int, num_envs: int, num_evals: int):
    """DeepMind's tuned PPO config for the G1, with our run length and width."""
    from mujoco_playground.config import locomotion_params

    p = locomotion_params.brax_ppo_config(BASE_ENV)
    p.num_timesteps, p.num_envs, p.num_evals = num_timesteps, num_envs, num_evals
    return p


def inference_fn(net_cfg: dict, params, env):
    import functools

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    net = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)(
        env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize)
    return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))


# {{docs-fragment rollout}}
def rollout(env, act, command, steps: int, seed: int = 0) -> dict:
    """Hold one command for ``steps`` and measure what the robot actually did."""
    import jax
    import jax.numpy as jp
    import numpy as np

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    feet = [env.mj_model.sensor_adr[s] for s in env._feet_floor_found_sensor]
    cmd = jp.array(command, dtype=jp.float32)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    state.info["command"] = cmd
    traj, speed, airborne, reward = [state], [], [], 0.0
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        state = step(state, act(state.obs, key))
        state.info["command"] = cmd  # the env would resample it mid-episode
        traj.append(state)
        reward += float(state.reward)
        speed.append(float(env.get_local_linvel(state.data, "pelvis")[0]))
        airborne.append(not any(float(state.data.sensordata[a]) > 0 for a in feet))
        if float(state.done) > 0.5:
            break
    start, end = np.asarray(traj[0].data.qpos[:2]), np.asarray(traj[-1].data.qpos[:2])
    settled = speed[len(speed) // 4:] or speed  # skip the first second of getting up to speed
    return {
        "traj": traj, "steps": len(traj) - 1, "reward": reward, "fell": float(state.done) > 0.5,
        "speed": float(np.mean(settled)) if settled else 0.0,
        "flight": float(np.mean(airborne[len(airborne) // 4:] or airborne)) if airborne else 0.0,
        "distance": float(np.linalg.norm(end - start)),
    }
# {{/docs-fragment rollout}}


def render_side(env, traj, height: int, width: int) -> list:
    """Film the robot in profile, running left to right.

    The camera follows the pelvis and turns with the robot's smoothed heading: reset randomizes
    the yaw and the policy drifts a little, so a fixed camera ends up behind it.
    """
    import math

    import mujoco
    import numpy as np

    m = env.mj_model
    renderer = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation = 3.0, -8.0
    d, frames, hx, hy = mujoco.MjData(m), [], None, None
    for state in traj:
        d.qpos, d.qvel = state.data.qpos, state.data.qvel
        mujoco.mj_forward(m, d)
        w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # Low-pass the heading so the gait's hip sway doesn't shake the camera.
        hx = math.cos(yaw) if hx is None else 0.95 * hx + 0.05 * math.cos(yaw)
        hy = math.sin(yaw) if hy is None else 0.95 * hy + 0.05 * math.sin(yaw)
        cam.azimuth = math.degrees(math.atan2(hy, hx)) + 90.0
        renderer.update_scene(d, camera=cam)
        frames.append(np.asarray(renderer.render()))
    renderer.close()
    return frames


def encode_mp4(frames, fps: int) -> bytes:
    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("libx264", rate=fps)
        h, w = np.asarray(frames[0]).shape[:2]
        s.width, s.height, s.pix_fmt, s.options = w - w % 2, h - h % 2, "yuv420p", {"crf": "26"}
        for f in frames:
            out.mux(s.encode(av.VideoFrame.from_ndarray(np.asarray(f, dtype=np.uint8)[: s.height, : s.width], format="rgb24")))
        out.mux(s.encode())
    return buf.getvalue()


def still(frame, width: int = 240) -> str:
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def video(mp4: bytes, label: str) -> str:
    return (f"<video controls autoplay loop muted playsinline style='width:100%;border-radius:8px;background:#000' "
            f"aria-label='{viz.esc(label)}' src='data:video/mp4;base64,{base64.b64encode(mp4).decode()}'></video>")

