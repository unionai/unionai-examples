"""E6 — training steps as separate Flyte tasks on ONE reusable Ray cluster.

Question: does Ray-side state (actors, driver connection) survive across Flyte task boundaries,
across a task that raises, and across a driver crash+retry — i.e. can each step be a durable
Flyte boundary while the cluster (and the model actors) stay warm?

    flyte --config ~/.flyte/demo-ketan.yaml run e6_reusable_ray_steps.py driver
"""

import asyncio
import os
import socket
import time

import flyte
import flyte.errors

try:
    import ray
    from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(),
        worker_node_config=[WorkerNodeConfig(group_name="w", replicas=2)],
        enable_autoscaling=False,
    )
except ImportError:  # local parse without the plugin
    ray = None
    ray_config = None

image = (
    flyte.Image.from_debian_base(name="skyrl-e6-ray")
    .with_apt_packages("wget")  # the reusable Ray head needs it for its readiness probe
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray")
    .with_local_v2()
)

ray_env = flyte.TaskEnvironment(
    name="skyrl-e6-ray",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("2000Mi", "4000Mi")),
    reusable=flyte.ReusePolicy(replicas=1, idle_ttl=600, scope="run"),
)
driver_env = flyte.TaskEnvironment(
    name="skyrl-e6-driver", image=image, resources=flyte.Resources(cpu=1, memory="1Gi"), depends_on=[ray_env]
)

_STATE: dict = {}  # process-global: survives across task invocations only if the replica process is reused


def _actor_cls():
    @ray.remote
    class Policy:
        def __init__(self):
            self.version = 0
            self.pid = os.getpid()

        def step(self):
            self.version += 1
            return self.version

        def info(self):
            return {"actor_pid": self.pid, "version": self.version}

    return Policy


def _get_policy():
    """Held in a module global (like an actor group cached per replica) AND created as a named
    detached actor, so we can tell apart 'process survived' from 'actor survived'."""
    if "policy" not in _STATE:
        try:  # re-attach if a previous task (a different Ray job) already created it
            _STATE["policy"] = ray.get_actor("policy", namespace="skyrl-e6")
            _STATE["created_in_pid"] = -1  # -1 = re-attached, not created here
            return _STATE["policy"]
        except ValueError:
            pass
        Policy = _actor_cls()
        _STATE["policy"] = Policy.options(
            name="policy", namespace="skyrl-e6", lifetime="detached", get_if_exists=True
        ).remote()
        _STATE["created_in_pid"] = os.getpid()
    return _STATE["policy"]


@ray_env.task(retries=1)
async def train_step(k: int, fail_on_first_try: bool = False) -> dict:
    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    policy = _get_policy()
    version = ray.get(policy.step.remote())
    info = ray.get(policy.info.remote())
    out = {
        "k": k,
        "attempt": attempt,
        "task_pid": os.getpid(),
        "task_host": socket.gethostname(),
        "ray_job_id": ray.get_runtime_context().get_job_id(),
        "actor_created_in_pid": _STATE.get("created_in_pid"),
        "policy_version": version,
        **info,
        "t": time.time(),
    }
    print("E6_STEP", out, flush=True)
    if fail_on_first_try and attempt == 0:
        raise RuntimeError(f"E6 simulated failure in step {k} attempt 0")
    return out


@driver_env.task(retries=1)
async def driver(n_steps: int = 6) -> list[dict]:
    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    results = []
    t0 = time.time()
    for k in range(n_steps):
        r = await train_step(k, fail_on_first_try=(k == 2))
        r["step_wall_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        results.append(r)
        if attempt == 0 and k == 3:
            raise flyte.errors.RuntimeSystemError("simulated", "E6 driver crash after step 3 on attempt 0")
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(driver)
    print(r.url)
