# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.15.0",
#    "kubernetes",
# ]
# ///

"""Test a KV cache pool shared by two Kimi K3 nodes, through Mooncake over TCP.

Two Flyte tasks, a and b, each get a whole MI355X node, copy the weights into
node memory and start vLLM with MooncakeStoreConnector. Node a also runs the
Mooncake master. Every GPU worker on both nodes adds part of its node's RAM to
one pool, so a prefix that either node computed can be loaded by the other
instead of being computed again.

Node a then measures time to first token for:

- a-cold: documents node a has never seen.
- b-from-pool: the same documents on node b, which can only find them in the pool.
- b-cold: new documents on node b, the baseline for b-from-pool.
- a-local: the first documents again on node a, which still has them in GPU memory.

With --b-configs, the test repeats for each set of environment variables:
node b restarts vLLM with them, and node a sends new documents, then b loads
them from the pool. Both nodes record their network throughput during each
phase, which shows whether loading from the pool is limited by the network.

Results, vLLM logs and the Mooncake master log go under the results prefix.
Both tasks exit when the test ends, which releases the nodes.

The Pensando NICs on these nodes cannot register a large host-memory pool for
RDMA, so the pool uses TCP. The tasks stay on the pod network, the same network
the serving app uses.

    uv run kv_pool_test.py --model <model path> --results <object store prefix>
"""

import argparse
import asyncio
import hashlib
import json
import os

import flyte
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from common import GPUS_PER_NODE, QUEUE
from kimi_app import CACHE_DIR, HF_REPO, SERVED_NAME, VLLM_BASE
from kimi_app import image as kimi_image

MOONCAKE = "mooncake-transfer-engine-rocm==0.3.14"
MC_DIR = "/opt/mc"
MASTER_PORT = 50051
VLLM_PORT = 8000

# The image's entrypoint is vllm. A small shim in front of the real binary
# passes Flyte's task command through: a0 runs as given, anything else goes to vllm.
SHIM_B64 = "IyEvYmluL3NoCmlmIFsgIiQxIiA9IGEwIF07IHRoZW4gZXhlYyAiJEAiOyBmaQpleGVjIC9vcHQvcnVudGltZS12ZW52L2Jpbi92bGxtLXJlYWwgIiRAIgo="
image = (
    kimi_image.clone(name="kv-pool", extendable=True)
    .with_commands(
        [
            "mv /opt/runtime-venv/bin/vllm /opt/runtime-venv/bin/vllm-real",
            f"echo {SHIM_B64} | base64 -d > /opt/runtime-venv/bin/vllm && chmod +x /opt/runtime-venv/bin/vllm",
            # Kept out of vLLM's virtualenv so its packages stay as built; vLLM finds it through PYTHONPATH.
            f"/opt/runtime-venv/bin/python -m pip install --no-deps --target {MC_DIR} {MOONCAKE}"
            f" || uv pip install --no-deps --python /opt/runtime-venv/bin/python --target {MC_DIR} {MOONCAKE}",
        ]
    )
    .with_pip_packages("flyte==2.15.0", "kubernetes")
)

# Without these, Mooncake's TCP transport uses one I/O thread and one
# connection per peer, which measured 1.5 GB/s between nodes instead of 11.
MC_TCP_ENV = {
    "MC_NUM_TCP_IO_THREADS": "16",
    "MC_TCP_ENABLE_CONNECTION_POOL": "1",
    "MC_TCP_LANES_PER_PEER": "8",
    "MC_TCP_MAX_CONNECTIONS_PER_PEER": "16",
}


def node_pod_template() -> flyte.PodTemplate:
    gpus = {"amd.com/gpu": str(GPUS_PER_NODE)}
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    resources=V1ResourceRequirements(requests=gpus, limits=gpus),
                    volume_mounts=[
                        V1VolumeMount(name="shm", mount_path="/dev/shm"),
                        V1VolumeMount(name="cache", mount_path=CACHE_DIR),
                    ],
                )
            ],
            volumes=[
                V1Volume(name="shm", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="128Gi")),
                V1Volume(name="cache", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="1700Gi")),
            ],
            tolerations=[V1Toleration(key="amd.com/gpu", operator="Exists", effect="NoSchedule")],
        ),
    )


node_env = flyte.TaskEnvironment(
    name="kv-pool-node",
    image=image,
    # The node has about 2.7 TiB: the weights volume, the pool share and vLLM's host memory.
    resources=flyte.Resources(cpu="48", memory="2500Gi"),
    pod_template=node_pod_template(),
    env_vars={"VLLM_ROCM_USE_AITER": "1"},
    queue=QUEUE,
)

driver_env = flyte.TaskEnvironment(
    name="kv-pool",
    # The driver only waits on the two node tasks, so it skips the large vLLM image.
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("kubernetes"),
    resources=flyte.Resources(cpu="1", memory="2Gi"),
    depends_on=[node_env],
    queue=QUEUE,
)

# Runs in vLLM's own Python, which has httpx and openai. Sends each phase's
# documents to one node and records time to first token, how many prompt
# tokens came from a cache, and how vLLM's cache counters moved.
CLIENT = r'''
import asyncio, json, random, sys, time

import httpx
from openai import AsyncOpenAI

cfg = json.loads(sys.argv[1])
SERVED = cfg["served_name"]
TOPICS = ["caching", "retries", "scheduling", "storage", "networking", "billing", "security", "logging"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_context(seed, target_words=6000):
    rng = random.Random(seed)
    lines, words, i = [], 0, 0
    while words < target_words:
        t = rng.choice(TOPICS)
        line = (f"Fact {i} ({t}): component {rng.randint(100, 999)} uses a {t} policy with limit "
                f"{rng.randint(1, 500)} and a timeout of {rng.randint(1, 60)} seconds, owned by team {rng.randint(1, 40)}.")
        lines.append(line); words += len(line.split()); i += 1
    return f"Reference document, version {seed}.\n" + "\n".join(lines) + "\n\nQuestion: which team owns fact 3?"


def metrics(base):
    out = {}
    try:
        text = httpx.get(f"{base}/metrics", timeout=20).text
    except Exception as e:
        return {"error": str(e)}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name = line.split(" ")[0] if "mooncake" in line else line.split("{")[0].split(" ")[0]
        if any(k in name for k in ("prefix_cache", "kv", "connector", "mooncake", "prompt_tokens")) and not name.endswith(("_bucket", "_created")):
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def saved_bytes(base):
    m = metrics(base)
    return sum(v for k, v in m.items() if "mooncake_store_operation_bytes_total" in k and "put" in k)


async def ttft(client, content):
    start = time.perf_counter()
    stream = await client.chat.completions.create(
        model=SERVED, messages=[{"role": "user", "content": content}], max_tokens=16, stream=True,
        stream_options={"include_usage": True})
    first, cached, prompt = None, None, None
    async for chunk in stream:
        d = chunk.choices[0].delta if chunk.choices else None
        if first is None and d is not None and (getattr(d, "content", None) or getattr(d, "reasoning_content", None)):
            first = time.perf_counter() - start
        if chunk.usage:
            prompt = chunk.usage.prompt_tokens
            det = getattr(chunk.usage, "prompt_tokens_details", None)
            cached = getattr(det, "cached_tokens", None) if det else None
    return (first if first is not None else time.perf_counter() - start), prompt, cached


def summary(xs):
    s = sorted(xs)
    return {"n": len(s), "p50_ms": round(s[len(s) // 2] * 1000), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))] * 1000),
            "mean_ms": round(sum(s) / len(s) * 1000)}


async def phase(name, base, contexts, concurrency):
    client = AsyncOpenAI(base_url=f"{base}/v1", api_key="none", timeout=900, max_retries=2)
    sem = asyncio.Semaphore(concurrency)

    async def one(c):
        async with sem:
            return await ttft(client, c)

    m0 = metrics(base)
    t0 = time.time()
    start = time.perf_counter()
    rs = await asyncio.gather(*(one(c) for c in contexts))
    wall = time.perf_counter() - start
    t1 = time.time()
    m1 = metrics(base)
    delta = {k: round(m1[k] - m0.get(k, 0.0), 1) for k in m1 if isinstance(m1.get(k), float) and m1[k] != m0.get(k, 0.0)}
    prompt = sum(r[1] or 0 for r in rs); cached = sum(r[2] or 0 for r in rs)
    row = {"phase": name, "node": base, "docs": len(contexts), "concurrency": concurrency, **summary([r[0] for r in rs]),
           "wall_s": round(wall, 1), "prompt_tokens": prompt, "cached_tokens": cached,
           "cached_fraction": round(cached / prompt, 3) if prompt else None, "start": t0, "end": t1,
           "mooncake": mooncake_ops(delta), "metric_deltas": delta}
    log(json.dumps({k: v for k, v in row.items() if k != "metric_deltas"}))
    return row


def mooncake_ops(delta):
    """Per operation: calls, total time across GPU workers, average time and bytes."""
    import re
    ops = {}
    for k, v in delta.items():
        g = re.search(r'operation="(\w+)",status="(\w+)"', k)
        if not g:
            continue
        op = ops.setdefault(f"{g.group(1)}/{g.group(2)}", {})
        for metric, key, scale in (("operation_time_seconds_sum", "time_s", 1), ("operation_time_seconds_count", "calls", 1),
                                   ("operation_bytes_total", "gb", 1e-9)):
            if k.startswith(f"vllm:mooncake_store_{metric}"):
                op[key] = round(v * scale, 2)
    for op in ops.values():
        if op.get("calls"):
            op["avg_ms"] = round(1000 * op.get("time_s", 0) / op["calls"], 1)
    return ops


def wait_for_saves(base, timeout=900):
    """Saves to the pool finish after the responses, so wait until the byte count stops growing."""
    last, still, start = -1.0, 0, time.time()
    while time.time() - start < timeout:
        b = saved_bytes(base)
        still = still + 1 if b == last else 0
        if still >= 3:
            break
        last = b
        time.sleep(10)
    log("saved bytes on", base, last, f"after {round(time.time() - start)}s")
    return {"saved_bytes": last, "waited_s": round(time.time() - start)}


async def main():
    a, b, n, conc, full = cfg["a"], cfg["b"], cfg["docs"], cfg["concurrency"], cfg["full"]
    seed = random.randint(0, 10**9)
    docs = [make_context(seed + i) for i in range(n)]
    fresh = [make_context(seed + n + i) for i in range(n)]
    # First requests compile kernels, so they stay out of the measured phases.
    # Later rounds only restart vLLM on b, so only b needs warming up again.
    await asyncio.gather(*(phase(f"warmup-{x}", x, [make_context(seed - 1 - i - 100 * j) for i in range(8)], 8)
                           for j, x in enumerate((a, b) if full else (b,))))
    rows = [await phase("a-cold", a, docs, conc)]
    saves = wait_for_saves(a)
    rows.append(await phase("b-from-pool", b, docs, conc))
    if full:
        rows.append(await phase("b-cold", b, fresh, conc))
        rows.append(await phase("a-local", a, docs, conc))
    json.dump({"rows": rows, "a_saves": saves}, open(cfg["out"], "w"), indent=1)


asyncio.run(main())
'''


class _Store:
    """Small JSON files under the results prefix, for the two tasks to find each other."""

    def __init__(self, res: str):
        self.res = res.rstrip("/")

    async def put(self, name: str, obj) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(obj, f, indent=1)
        await flyte.storage.put(f.name, f"{self.res}/{name}")

    async def get(self, name: str):
        path = f"{self.res}/{name}"
        try:
            flyte.storage.get_underlying_filesystem(path=path).invalidate_cache()
        except Exception:
            pass
        if not await flyte.storage.exists(path):
            return None
        with open(await flyte.storage.get(path)) as f:
            return json.load(f)

    async def wait(self, name: str, timeout: int, alive=None, abort: str | None = None):
        import time

        start = time.time()
        while time.time() - start < timeout:
            got = await self.get(name)
            if got is not None:
                return got
            if alive is not None and not alive():
                raise RuntimeError(f"stopped waiting for {name}: local vLLM exited")
            if abort and (failed := await self.get(abort)) is not None:
                raise RuntimeError(f"stopped waiting for {name}: {abort} says {failed}")
            await asyncio.sleep(15)
        raise TimeoutError(f"{name} did not appear within {timeout}s")


def _net_sampler(samples: list, stop) -> None:
    """Every second, record bytes received and sent on the pod's network interfaces."""
    import time

    while not stop.is_set():
        rx = tx = 0
        with open("/proc/net/dev") as f:
            for line in f.readlines()[2:]:
                name, data = line.split(":", 1)
                if name.strip() == "lo":
                    continue
                v = data.split()
                rx += int(v[0])
                tx += int(v[8])
        samples.append((time.time(), rx, tx))
        stop.wait(1)


def _net_rates(samples: list, start: float, end: float) -> dict:
    """Average receive and send rates in GB/s between two times."""
    win = [s for s in samples if start <= s[0] <= end]
    if len(win) < 2:
        return {}
    dt = win[-1][0] - win[0][0]
    return {"rx_gbps": round((win[-1][1] - win[0][1]) / dt / 1e9, 2), "tx_gbps": round((win[-1][2] - win[0][2]) / dt / 1e9, 2)}


@node_env.task(timeout=4 * 3600)
async def node(
    role: str,
    model: str,
    compile_remote: str,
    results: str,
    docs: int,
    concurrency: int,
    segment_gb: int,
    vllm_extra: list[str],
    b_configs: list[dict[str, str]],
) -> str:
    import shlex
    import signal
    import socket
    import subprocess
    import threading
    import time
    import urllib.request

    res = results.rstrip("/")
    store = _Store(res)
    ip = socket.gethostbyname(socket.gethostname())
    print(f"{role} on {socket.gethostname()} {ip}", flush=True)
    samples: list = []
    stop_sampling = threading.Event()
    threading.Thread(target=_net_sampler, args=(samples, stop_sampling), daemon=True).start()

    def ts(event: str) -> None:
        print(f"STARTUP {int(time.time())} {role} {event}", flush=True)

    env = dict(os.environ)
    mc_libs = f"{MC_DIR}/mooncake:{MC_DIR}/mooncake_transfer_engine_rocm.libs"
    env.update(
        AWS_ACCESS_KEY_ID=env.get("FLYTE_AWS_ACCESS_KEY_ID", ""),
        AWS_SECRET_ACCESS_KEY=env.get("FLYTE_AWS_SECRET_ACCESS_KEY", ""),
        AWS_REGION="auto",
    )
    s5 = f"s5cmd --endpoint-url {shlex.quote(env.get('FLYTE_AWS_ENDPOINT', ''))} --numworkers 64 --log error"

    def sh(cmd: str, check: bool = True) -> int:
        rc = subprocess.run(["bash", "-c", cmd], env=env).returncode
        if check and rc:
            raise RuntimeError(f"failed ({rc}): {cmd}")
        return rc

    def upload(path: str) -> None:
        sh(f"{s5} cp {shlex.quote(path)} {shlex.quote(res)}/{role}/", check=False)

    procs: list[subprocess.Popen] = []
    vllm: dict = {}
    out_dir = f"{CACHE_DIR}/out"

    def stop_vllm() -> None:
        p = vllm.get("proc")
        if p is None:
            return
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                pass
        # vLLM runs in its own session, so this also reaches workers left holding GPU memory.
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        vllm.clear()

    def finish() -> None:
        stop_vllm()
        for p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(5)
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stop_sampling.set()
        with open(f"{out_dir}/net.json", "w") as f:
            json.dump(samples, f)
        sh(f"free -g > {out_dir}/free.txt", check=False)
        for name in sorted(os.listdir(out_dir)):
            if name.endswith((".json", ".txt", ".log")) and not name.startswith("vllm-"):
                upload(f"{out_dir}/{name}")
        for name in sorted(os.listdir(out_dir)):
            if name.startswith("vllm-") and name.endswith(".log"):
                sh(f"tail -c 2000000 {out_dir}/{name} > {out_dir}/tail-{name}", check=False)
                upload(f"{out_dir}/tail-{name}")

    try:
        ts("copy-start")
        w = f"{CACHE_DIR}/weights"
        sh(f"mkdir -p {w} {CACHE_DIR}/compile {out_dir}")
        if sh(f"{s5} cp --concurrency 8 --part-size 64 {shlex.quote(model.rstrip('/'))}/* {w}/", check=False):
            ts("copy-failed-falling-back-to-hugging-face")
            sh(f"hf download {HF_REPO} --local-dir {w} --max-workers 16")
        sh(f"{s5} cp {shlex.quote(compile_remote.rstrip('/'))}/* {CACHE_DIR}/compile/ >/dev/null 2>&1", check=False)
        ts("copy-done")

        if role == "a":
            mlog = open(f"{out_dir}/mooncake_master.log", "w")
            procs.append(
                subprocess.Popen(
                    [f"{MC_DIR}/mooncake/mooncake_master", f"--port={MASTER_PORT}"],
                    stdout=mlog, stderr=subprocess.STDOUT, start_new_session=True, cwd=f"{MC_DIR}/mooncake",
                    env={**env, "LD_LIBRARY_PATH": f"{mc_libs}:{env.get('LD_LIBRARY_PATH', '')}"},
                )
            )
            await asyncio.sleep(5)
            if procs[0].poll() is not None:
                raise RuntimeError(f"mooncake_master exited with {procs[0].returncode}")
            await store.put("master.json", {"addr": f"{ip}:{MASTER_PORT}"})
            master = f"{ip}:{MASTER_PORT}"
        else:
            master = (await store.wait("master.json", timeout=3 * 3600, abort="DONE.json"))["addr"]

        mc_cfg = f"{CACHE_DIR}/mooncake.json"
        with open(mc_cfg, "w") as f:
            json.dump(
                {
                    "mode": "embedded",
                    "metadata_server": "P2PHANDSHAKE",
                    "master_server_address": master,
                    # Per GPU worker, so the node adds eight times this to the pool.
                    "global_segment_size": f"{segment_gb}GB",
                    "local_buffer_size": "4GB",
                    "protocol": "tcp",
                    "device_name": "",
                    "enable_offload": False,
                },
                f,
            )
        base_env = dict(env)
        base_env.update(MC_TCP_ENV)
        base_env.update(
            PYTHONPATH=f"{MC_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":"),
            LD_LIBRARY_PATH=f"{mc_libs}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":"),
            MOONCAKE_CONFIG_PATH=mc_cfg,
            VLLM_HOST_IP=ip,
            # Both nodes must hash prompts the same way to find each other's blocks.
            PYTHONHASHSEED="0",
            HF_HUB_OFFLINE="1",
            VLLM_CACHE_ROOT=f"{CACHE_DIR}/compile/vllm",
            TORCHINDUCTOR_CACHE_DIR=f"{CACHE_DIR}/compile/inductor",
            TRITON_CACHE_DIR=f"{CACHE_DIR}/compile/triton",
        )
        rc = subprocess.run(
            ["/opt/runtime-venv/bin/python", "-c", "from mooncake.store import MooncakeDistributedStore"], env=base_env
        ).returncode
        if rc:
            raise RuntimeError("vLLM's Python cannot import mooncake.store")

        kv_cfg = {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both", "kv_load_failure_policy": "recompute"}
        args = [
            "/opt/runtime-venv/bin/vllm-real", "serve", w,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--tensor-parallel-size", str(GPUS_PER_NODE),
            "--trust-remote-code",
            "--reasoning-parser", "kimi_k3",
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.92",
            "--kv-transfer-config", json.dumps(kv_cfg),
            *vllm_extra,
        ]

        async def start_vllm(tag: str, extra_env: dict[str, str]) -> None:
            path = f"{out_dir}/vllm-{tag}.log"
            ts(f"vllm-start {tag} {extra_env}")
            p = subprocess.Popen(
                args, stdout=open(path, "w"), stderr=subprocess.STDOUT, env={**base_env, **extra_env}, start_new_session=True
            )
            vllm.update(proc=p, log=path)
            deadline = time.time() + 2400
            while True:
                if p.poll() is not None:
                    with open(path, errors="replace") as f:
                        raise RuntimeError(f"vllm exited with {p.returncode}:\n{f.read()[-4000:]}")
                try:
                    if urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5).status == 200:
                        break
                except Exception:
                    pass
                if time.time() > deadline:
                    raise RuntimeError("vllm not ready after 40 minutes")
                await asyncio.sleep(10)
            ts(f"vllm-ready {tag}")

        def alive() -> bool:
            return vllm.get("proc") is not None and vllm["proc"].poll() is None

        if role == "b":
            # One round per config: a names the round, b restarts vLLM with its environment.
            for i in range(len(b_configs)):
                got = await store.wait(f"round-{i}.json", timeout=3 * 3600, abort="DONE.json")
                stop_vllm()
                await start_vllm(f"round{i}", got["env"])
                await store.put(f"ready-b-{i}.json", {"url": f"http://{ip}:{VLLM_PORT}", "ts": time.time()})
            await store.wait("DONE.json", timeout=3 * 3600, alive=alive)
            windows = (await store.get("windows.json")) or []
            return json.dumps([{"role": "b", **w, **_net_rates(samples, w["start"], w["end"])} for w in windows])

        await start_vllm("a", {})
        rows = []
        for i, extra_env in enumerate(b_configs):
            await store.put(f"round-{i}.json", {"env": extra_env})
            peer = await store.wait(f"ready-b-{i}.json", timeout=3 * 3600, alive=alive, abort="failed-b.json")
            out = f"{out_dir}/result-{i}.json"
            client_cfg = {
                "served_name": SERVED_NAME,
                "a": f"http://{ip}:{VLLM_PORT}",
                "b": peer["url"],
                "docs": docs,
                "concurrency": concurrency,
                "full": i == 0,
                "out": out,
            }
            sh(f"/opt/runtime-venv/bin/python -u -c {shlex.quote(CLIENT)} {shlex.quote(json.dumps(client_cfg))}")
            with open(out) as f:
                doc = json.load(f)
            for r in doc["rows"]:
                r = {k: v for k, v in r.items() if k != "metric_deltas"}
                rows.append({"round": i, "b_env": extra_env, **r, "a_net": _net_rates(samples, r["start"], r["end"])})
            rows.append({"round": i, "a_saves": doc["a_saves"]})
            await store.put("rows.json", rows)
        await store.put(
            "windows.json",
            [{"round": r["round"], "phase": r["phase"], "start": r["start"], "end": r["end"]} for r in rows if "phase" in r],
        )
        return json.dumps(rows)
    except Exception as e:
        if role == "b":
            await store.put("failed-b.json", {"error": repr(e)[-2000:]})
        raise
    finally:
        if role == "a":
            await store.put("DONE.json", {"ts": time.time()})
        finish()


@driver_env.task
async def pool_test(
    model: str,
    compile_remote: str,
    results: str,
    docs: int = 200,
    concurrency: int = 32,
    segment_gb: int = 48,
    vllm_extra: list[str] = [],
    b_configs: list[dict[str, str]] = [{}],
) -> list[str]:
    args = (model, compile_remote, results, docs, concurrency, segment_gb, vllm_extra, b_configs)
    return list(await asyncio.gather(node("a", *args), node("b", *args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Remote path of the prefetched model directory")
    parser.add_argument("--results", required=True, help="Object store prefix for results and logs")
    parser.add_argument("--compile-cache", default=None, help="Object store prefix of the compile cache (default: the serving app's)")
    parser.add_argument("--docs", type=int, default=200, help="Distinct ~8k-token documents per phase")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--segment-gb", type=int, default=48, help="Pool memory per GPU worker, in GB")
    # The image's MoE and attention kernels fail with hipErrorLaunchOutOfResources
    # on some batches between about 14k and 16k tokens, so batches stay at 8k.
    parser.add_argument("--vllm-args", default="--max-num-batched-tokens 8192", help="Extra flags for vllm serve")
    parser.add_argument(
        "--b-configs",
        default="[{}]",
        help="JSON list of extra environment variables for vLLM on b, one test round each",
    )
    a = parser.parse_args()
    # Same key the serving app uses with no extra flags.
    key = hashlib.sha256(f"{VLLM_BASE} ".encode()).hexdigest()[:12]
    compile_remote = a.compile_cache or f"{a.model.rstrip('/')}-compile-cache/{key}"
    flyte.init_from_config()
    run = flyte.with_runcontext(queue=QUEUE).run(
        pool_test, model=a.model, compile_remote=compile_remote, results=a.results,
        docs=a.docs, concurrency=a.concurrency, segment_gb=a.segment_gb, vllm_extra=a.vllm_args.split(),
        b_configs=json.loads(a.b_configs),
    )
    print(run.name, run.url)
