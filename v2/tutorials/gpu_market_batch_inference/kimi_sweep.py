# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# ///

"""Compare vLLM configurations for Kimi K3 on one node without redeploying.

Each new deployment of the serving app costs a node grant and, after it ends,
a round of node checks. This app instead copies the weights into node memory
once, then for each configuration starts vLLM, runs the KV cache working-set
test against it from inside the pod, and stops it again. The vLLM log, the
results and the cache metrics of every configuration go to object storage as
soon as that configuration finishes, so a crash loses one configuration and
leaves its log behind.

A small status server on the app port stays up the whole time, so the app
stays healthy while vLLM restarts, and serves the results directory.

    uv run kimi_sweep.py --model <model path> --results <object store prefix>
"""

import argparse
import json
import os

import flyte
import flyte.app
from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1PodSpec,
    V1ResourceRequirements,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from common import GPUS_PER_NODE
from kimi_app import CACHE_DIR, HF_REPO, SERVED_NAME, image

STATUS_PORT = 8080
VLLM_PORT = 8000

# name -> extra vllm serve flags. Runs in this order.
CONFIGS = {
    "baseline": [],
    "cpu-offload-300": ["--kv-offloading-size", "300"],
    "batched-16k": ["--max-num-batched-tokens", "16384"],
}

# Copy the weights and compile caches into node memory, then hand over to the driver.
PRELUDE = r"""
set -euo pipefail
MODEL="$1"; HF_REPO="$2"; CACHE="$3"; COMPILE_REMOTE="$4"; DRIVER="$5"; shift 5
export AWS_ACCESS_KEY_ID="$FLYTE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$FLYTE_AWS_SECRET_ACCESS_KEY" AWS_REGION=auto
S5="s5cmd --endpoint-url $FLYTE_AWS_ENDPOINT --numworkers 64 --log error"
mkdir -p "$CACHE/weights" "$CACHE/compile" "$CACHE/results"
# Answer the app's health check right away; the driver takes this port over later.
(cd "$CACHE/results" && exec /opt/runtime-venv/bin/python -m http.server "$SERVE_PORT") &
STATUS_PID=$!
echo "STARTUP $(date +%s) copy-start"
if [ ! -f "$CACHE/weights/.complete" ]; then
  $S5 cp --concurrency 8 --part-size 64 "${MODEL%/}/*" "$CACHE/weights/" \
    || hf download "$HF_REPO" --local-dir "$CACHE/weights" --max-workers 16
  echo -n "$MODEL" > "$CACHE/weights/.complete"
fi
$S5 cp "${COMPILE_REMOTE%/}/*" "$CACHE/compile/" >/dev/null 2>&1 || true
echo "STARTUP $(date +%s) copy-done"
kill "$STATUS_PID" || true
printf '%s' "$DRIVER" > /tmp/sweep_driver.py
exec /opt/runtime-venv/bin/python -u /tmp/sweep_driver.py "$@"
"""

# Runs in the image's own Python, which has vLLM's dependencies (httpx, openai).
DRIVER = r'''
import asyncio, http.server, json, os, random, signal, socketserver, subprocess, sys, threading, time

import httpx
from openai import AsyncOpenAI

cfg = json.loads(sys.argv[1])
CACHE, RESULTS_REMOTE = cfg["cache"], cfg["results_remote"].rstrip("/")
OUT = f"{CACHE}/results"
VLLM_PORT, SERVED = cfg["vllm_port"], cfg["served_name"]
BASE = f"http://localhost:{VLLM_PORT}"
S5 = ["s5cmd", "--endpoint-url", os.environ["FLYTE_AWS_ENDPOINT"], "--log", "error"]
TOPICS = ["caching", "retries", "scheduling", "storage", "networking", "billing", "security", "logging"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def status_server():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=OUT, **k)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(("0.0.0.0", int(os.environ["SERVE_PORT"])), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def upload(path):
    subprocess.run([*S5, "cp", path, f"{RESULTS_REMOTE}/"], check=False)


def make_context(seed, target_words=6000):
    rng = random.Random(seed)
    lines, words, i = [], 0, 0
    while words < target_words:
        t = rng.choice(TOPICS)
        line = (f"Fact {i} ({t}): component {rng.randint(100, 999)} uses a {t} policy with limit "
                f"{rng.randint(1, 500)} and a timeout of {rng.randint(1, 60)} seconds, owned by team {rng.randint(1, 40)}.")
        lines.append(line); words += len(line.split()); i += 1
    return f"Reference document, version {seed}.\n" + "\n".join(lines)


def metrics():
    """Counters whose names mention the prefix cache, KV or offloading, summed over labels."""
    out = {}
    try:
        text = httpx.get(f"{BASE}/metrics", timeout=20).text
    except Exception as e:
        return {"error": str(e)}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(k in name for k in ("prefix_cache", "kv", "offload", "connector", "prompt_tokens")) and not name.endswith(("_bucket", "_created")):
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


async def ttft(client, content, max_tokens=16):
    start = time.perf_counter()
    stream = await client.chat.completions.create(
        model=SERVED, messages=[{"role": "user", "content": content}], max_tokens=max_tokens, stream=True)
    first = None
    async for chunk in stream:
        d = chunk.choices[0].delta if chunk.choices else None
        if first is None and d is not None and (getattr(d, "content", None) or getattr(d, "reasoning_content", None)):
            first = time.perf_counter() - start
    return first if first is not None else time.perf_counter() - start


def summary(xs):
    s = sorted(xs)
    return {"n": len(s), "p50_ms": round(s[len(s) // 2] * 1000), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))] * 1000)}


async def working_set(docs, concurrency):
    """Send `docs` distinct ~8k-token documents, then the same ones again in order."""
    client = AsyncOpenAI(base_url=f"{BASE}/v1", api_key=os.environ["VLLM_API_KEY"], timeout=900, max_retries=2)
    base = random.randint(0, 10**9)
    contexts = [make_context(base + d) for d in range(docs)]
    rows = []
    # A short warmup so first-request compilation does not land in the first pass.
    await asyncio.gather(*(ttft(client, make_context(base - 1 - i)) for i in range(8)))
    for name in ("first-pass", "second-pass"):
        sem = asyncio.Semaphore(concurrency)
        m0 = metrics()

        async def one(d):
            async with sem:
                return await ttft(client, contexts[d] + "\n\nQuestion: which team owns fact 3?")

        start = time.perf_counter()
        xs = await asyncio.gather(*(one(d) for d in range(docs)))
        m1 = metrics()
        delta = {k: round(m1[k] - m0.get(k, 0.0), 1) for k in m1 if isinstance(m1.get(k), float) and m1[k] != m0.get(k, 0.0)}
        hits = delta.get("vllm:prefix_cache_hits_total", 0.0); queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
        row = {"pass": name, "docs": docs, "concurrency": concurrency, **summary(xs),
               "wall_s": round(time.perf_counter() - start, 1),
               "gpu_prefix_hit_rate": round(hits / queries, 3) if queries else None, "metric_deltas": delta}
        log(json.dumps({k: v for k, v in row.items() if k != "metric_deltas"}))
        rows.append(row)
    return rows


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    # vLLM runs in its own session, so this also reaches any worker left behind,
    # which would otherwise keep holding GPU memory.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(30)


def run_config(name, extra):
    logpath = f"{OUT}/{name}.log"
    args = ["/opt/runtime-venv/bin/vllm", "serve", f"{CACHE}/weights", "--port", str(VLLM_PORT), *cfg["vllm_args"], *extra]
    log("START", name, " ".join(extra))
    result = {"config": name, "extra_args": extra, "started": time.time()}
    with open(logpath, "w") as f:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.time() + 2400
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm exited with code {proc.returncode} before it was ready")
            try:
                if httpx.get(f"{BASE}/health", timeout=5).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(10)
        else:
            raise RuntimeError("vllm was not ready after 40 minutes")
        result["ready_s"] = round(time.time() - result["started"])
        log("READY", name, f"{result['ready_s']}s")
        result["rows"] = asyncio.run(working_set(cfg["docs"], cfg["concurrency"]))
        if proc.poll() is not None:
            result["error"] = f"vllm exited with code {proc.returncode} during the test"
    except Exception as e:
        result["error"] = repr(e)
        log("FAILED", name, repr(e))
    finally:
        result["free_mem"] = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        stop(proc)
        with open(f"{OUT}/{name}.json", "w") as f:
            json.dump(result, f, indent=1)
        upload(logpath); upload(f"{OUT}/{name}.json")
        log("DONE", name)
    return result


status_server()
os.environ.update({"HF_HUB_OFFLINE": "1", "VLLM_CACHE_ROOT": f"{CACHE}/compile/vllm",
                   "TORCHINDUCTOR_CACHE_DIR": f"{CACHE}/compile/inductor", "TRITON_CACHE_DIR": f"{CACHE}/compile/triton"})
summary_rows = []
for name, extra in cfg["configs"].items():
    r = run_config(name, extra)
    summary_rows.append({"config": name, "ready_s": r.get("ready_s"), "error": r.get("error"),
                         "passes": [{k: v for k, v in row.items() if k != "metric_deltas"} for row in r.get("rows", [])]})
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary_rows, f, indent=1)
    upload(f"{OUT}/summary.json")
open(f"{OUT}/ALL_DONE", "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
upload(f"{OUT}/ALL_DONE")
log("ALL DONE")
while True:
    time.sleep(3600)
'''


def sweep_app(model: str, compile_remote: str, results_remote: str, docs: int, concurrency: int) -> flyte.app.AppEnvironment:
    gpus = {"amd.com/gpu": str(GPUS_PER_NODE)}
    pod_spec = V1PodSpec(
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
    )
    vllm_args = [
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--tensor-parallel-size", str(GPUS_PER_NODE),
        "--trust-remote-code",
        "--reasoning-parser", "kimi_k3",
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
    ]
    driver_cfg = {
        "cache": CACHE_DIR,
        "results_remote": results_remote,
        "vllm_port": VLLM_PORT,
        "served_name": SERVED_NAME,
        "vllm_args": vllm_args,
        "configs": CONFIGS,
        "docs": docs,
        "concurrency": concurrency,
    }
    return flyte.app.AppEnvironment(
        name=os.getenv("BATCH_APP_NAME", "kimi-sweep"),
        image=image,
        command=[
            "bash", "-c", PRELUDE, "sweep", model, HF_REPO, CACHE_DIR, compile_remote, DRIVER, json.dumps(driver_cfg),
        ],
        port=STATUS_PORT,
        # The node has about 2.7 TiB. This covers the weights volume, vLLM's host
        # memory and a CPU KV offload buffer of a few hundred GiB.
        resources=flyte.Resources(cpu="48", memory="2500Gi", disk="200Gi"),
        env_vars={"VLLM_ROCM_USE_AITER": "1", "SERVE_PORT": str(STATUS_PORT)},
        secrets=flyte.Secret(key="kimi-vllm-api-key", as_env_var="VLLM_API_KEY"),
        scaling=flyte.app.Scaling(replicas=(1, 1)),
        requires_auth=False,
        cluster_pool=os.getenv("BATCH_CLUSTER_POOL"),
        pod_template=flyte.PodTemplate(
            pod_spec=pod_spec,
            annotations={"serving.knative.dev/progress-deadline": "10800s"},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Remote path of the prefetched model directory")
    parser.add_argument("--results", required=True, help="Object store prefix for logs and results")
    parser.add_argument("--compile-cache", required=True, help="Object store prefix of the compile cache")
    parser.add_argument("--docs", type=int, default=700, help="Distinct ~8k-token documents in the working set")
    parser.add_argument("--concurrency", type=int, default=32)
    a = parser.parse_args()

    flyte.init_from_config()
    app = flyte.serve(sweep_app(a.model, a.compile_cache, a.results, a.docs, a.concurrency))
    print(app.url if hasattr(app, "url") else app)
