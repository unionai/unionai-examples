"""E1 (warm-pool + small-image cold start only; big-image variant is e1_dispatch_latency.py)

E1 — per-action overhead on a reusable warm pool, and cold-start of fresh pods.

Answers: "how small can a rollout unit be before Flyte action overhead dominates?"
and "what does a fresh pod on a multi-GB image cost?" (the Flyte-pod-as-sandbox number).

Run (CPU only):
    flyte --config ~/.flyte/demo-config.yaml --project ketan --domain development \
        run e1_dispatch_latency.py main --n-warm 500 --n-cold 10 --n-big 3
"""

import asyncio
import json
import statistics
import time

import flyte
import flyte.report

small = flyte.Image.from_debian_base(name="skyrl-e1").with_pip_packages("unionai-reuse")
# A deliberately large image (vLLM + CUDA userspace, ~10 GB) to measure pull-dominated cold start.

pool_env = flyte.TaskEnvironment(
    name="skyrl-e1-pool",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    reusable=flyte.ReusePolicy(replicas=(2, 8), idle_ttl=120, concurrency=16, scaledown_ttl=120),
    image=small,
)
cold_env = flyte.TaskEnvironment(
    name="skyrl-e1-cold", resources=flyte.Resources(cpu=1, memory="1Gi"), image=small
)
driver_env = flyte.TaskEnvironment(
    name="skyrl-e1-driver",
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    image=small,
    depends_on=[pool_env, cold_env],
)


@pool_env.task
async def warm_noop(i: int) -> int:
    return i


@cold_env.task
async def cold_noop(i: int) -> int:
    return i


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {}
    xs = sorted(xs)
    q = statistics.quantiles(xs, n=20) if len(xs) >= 20 else xs
    return {
        "n": len(xs),
        "p50_s": round(statistics.median(xs), 3),
        "p95_s": round(q[-1] if len(xs) >= 20 else xs[-1], 3),
        "max_s": round(xs[-1], 3),
        "min_s": round(xs[0], 3),
    }


async def _timed(coro) -> float:
    t0 = time.monotonic()
    await coro
    return time.monotonic() - t0


@driver_env.task(report=True)
async def main(n_warm: int = 500, n_cold: int = 10, batch: int = 100) -> dict:
    out: dict = {}

    # --- warm pool: per-child wall time as seen by the parent (dispatch + queue + result) ---
    # First a primer batch so the pool is scaled up, then the measured batches.
    await asyncio.gather(*[warm_noop(i) for i in range(16)])
    lat: list[float] = []
    t_all = time.monotonic()
    for b0 in range(0, n_warm, batch):
        lat.extend(await asyncio.gather(*[_timed(warm_noop(i)) for i in range(b0, min(b0 + batch, n_warm))]))
    warm_wall = time.monotonic() - t_all
    out["warm_pool"] = {**_stats(lat), "batch": batch, "total_wall_s": round(warm_wall, 2),
                        "sustained_actions_per_s": round(n_warm / warm_wall, 2)}

    # --- cold: fresh pod per action on the small image (scheduling + start, image cached after 1st) ---
    cold = await asyncio.gather(*[_timed(cold_noop(i)) for i in range(n_cold)])
    out["cold_small_image"] = _stats(list(cold))


    flyte.report.log(f"<pre>{json.dumps(out, indent=2)}</pre>", do_flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
