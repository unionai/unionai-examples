"""E3 — what a Flyte retry of a Ray-plugin task actually does.

Crashes on attempt 0, succeeds on attempt 1, and records enough identity to tell whether attempt 1
ran on a fresh KubeRay cluster (different head/worker node ids) and where the task body ran
(head pod vs submitter). Attempt timestamps give the respin cost.

Run (CPU only):
    flyte --config <your-config> run e3_ray_retry.py crash_then_succeed
"""

import json
import os
import socket
import time

import flyte

ray_config = None
try:
    from flyteplugins.ray.task import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

    ray_config = RayJobConfig(
        head_node_config=HeadNodeConfig(),
        worker_node_config=[WorkerNodeConfig(group_name="w", replicas=1)],
        enable_autoscaling=False,
        shutdown_after_job_finishes=True,
        ttl_seconds_after_finished=120,
    )
except ImportError:  # local import for `flyte run` parsing without the plugin installed
    pass

image = (
    flyte.Image.from_debian_base(name="skyrl-e3-ray")
    .with_pip_packages("ray[default]==2.46.0", "flyteplugins-ray")
)

ray_env = flyte.TaskEnvironment(
    name="skyrl-e3-ray",
    plugin_config=ray_config,
    image=image,
    resources=flyte.Resources(cpu=(1, 2), memory=("1Gi", "2Gi")),
)


@ray_env.task(retries=2)
async def crash_then_succeed() -> dict:
    import ray

    attempt = int(os.environ.get("FLYTE_ATTEMPT_NUMBER", "0"))
    nodes = [
        {"id": n.get("NodeID", "")[:12], "host": n.get("NodeManagerHostname"), "alive": n.get("Alive")}
        for n in ray.nodes()
    ]
    info = {
        "attempt": attempt,
        "task_hostname": socket.gethostname(),
        "task_pod_ip": os.getenv("MY_POD_IP"),
        "ray_address": os.getenv("RAY_ADDRESS"),
        "ray_version": ray.__version__,
        "is_head_process": os.getenv("RAY_HEAD_SERVICE_HOST") is not None,
        "ray_nodes": nodes,
        "t_body_start": time.time(),
    }
    print("E3_INFO " + json.dumps(info), flush=True)
    if attempt == 0:
        raise RuntimeError("E3 simulated crash on attempt 0")
    return info


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(crash_then_succeed)
    print(r.url)
