# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# ///

"""Serve Kimi K3 on one MI355X node, loading its weights from node memory and reusing compiled kernels.

The weights are about 1.5 TB. At startup the app copies them from object
storage into a memory-backed volume with many parallel reads, and vLLM loads
them from there. If the copy from object storage fails, it falls back to
downloading from Hugging Face. A container restart in the same pod finds the
weights in place and skips the copy.

vLLM's compile caches (vLLM, Torch Inductor and Triton) are kept in object
storage. A new replica downloads them before starting vLLM, and once the
server is up it uploads what it compiled, so later replicas on any node skip
most of the kernel compilation in engine startup.

The vLLM ROCm build that supports Kimi K3 sets `vllm` as the image entrypoint
and keeps vLLM in its own virtualenv, /opt/runtime-venv. The app runs its own
command, which overrides the entrypoint.

    BATCH_VLLM_BASE=<vllm rocm image with Kimi K3> uv run kimi_app.py --model <model path>
"""

import argparse
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

VLLM_BASE = os.getenv("BATCH_VLLM_BASE", "docker.io/vllm/vllm-openai-rocm:kimi-k31-5c14fe8")
HF_REPO = os.getenv("BATCH_HF_REPO", "moonshotai/Kimi-K3")
S5CMD_VERSION = "2.3.0"
S5CMD_URL = (
    f"https://github.com/peak/s5cmd/releases/download/v{S5CMD_VERSION}/s5cmd_{S5CMD_VERSION}_Linux-64bit.tar.gz"
)
PORT = 8080
SERVED_NAME = "kimi-k3"
# A memory-backed volume for the weights and compile caches. The node's root
# disk is far slower: loading the weights from it took 12 minutes against about
# one from memory. The volume counts toward the pod's memory and lasts as long
# as the pod, so container restarts keep the weights.
CACHE_DIR = "/cache"
CACHE_SIZE = "1700Gi"

# s5cmd copies many files at once and splits each file into parallel parts.
image = (
    flyte.Image.from_base(VLLM_BASE)
    .clone(name="kimi-vllm", python_version=(3, 12), extendable=True)
    .with_commands(
        [
            # No quotes: the builder passes the command through a shell that drops them.
            f"(curl -fsSL {S5CMD_URL} || wget -qO- {S5CMD_URL}) | tar -xz -C /usr/local/bin s5cmd"
            " && chmod +x /usr/local/bin/s5cmd && /usr/local/bin/s5cmd version"
        ]
    )
)

# Copies the weights into the memory-backed volume, then starts vLLM on them.
# Each step prints a timestamp, so the log gives the startup breakdown.
SERVE = r"""
set -euo pipefail
MODEL="$1"; HF_REPO="$2"; CACHE="$3"; COMPILE_REMOTE="$4"; shift 4
ts() { echo "STARTUP $(date +%s) $*"; }
ts container-start
echo "RDMA devices: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"
export AWS_ACCESS_KEY_ID="$FLYTE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$FLYTE_AWS_SECRET_ACCESS_KEY" AWS_REGION=auto
S5="s5cmd --endpoint-url $FLYTE_AWS_ENDPOINT --numworkers 64 --log error"
DEST="$CACHE/weights"
COMPILE="$CACHE/compile"
mkdir -p "$DEST" "$COMPILE"
df -h "$CACHE" | sed 's/^/DISK /'
if [ -f "$DEST/.complete" ] && [ "$(cat "$DEST/.complete")" = "$MODEL" ]; then
  ts weights-already-present
else
  rm -f "$DEST/.complete"
  ts copy-start
  if $S5 cp --concurrency 8 --part-size 64 "${MODEL%/}/*" "$DEST/"; then
    ts copy-done-object-storage
  else
    ts copy-failed-falling-back-to-hugging-face
    hf download "$HF_REPO" --local-dir "$DEST" --max-workers 16
    ts copy-done-hugging-face
  fi
  du -sh "$DEST" | sed 's/^/WEIGHTS /'
  echo -n "$MODEL" > "$DEST/.complete"
fi
# Compiled kernels from earlier replicas, if any. A miss only means a slower start.
$S5 cp "${COMPILE_REMOTE%/}/*" "$COMPILE/" >/dev/null 2>&1 || true
du -sh "$COMPILE" | sed 's/^/COMPILE-CACHE /'
ts compile-cache-ready
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="$COMPILE/vllm" TORCHINDUCTOR_CACHE_DIR="$COMPILE/inductor" TRITON_CACHE_DIR="$COMPILE/triton"
# Once the server answers, upload what this replica compiled for the next one.
(
  until /opt/runtime-venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:$SERVE_PORT/health', timeout=5)" >/dev/null 2>&1; do sleep 10; done
  ts server-ready
  $S5 sync "$COMPILE/" "${COMPILE_REMOTE%/}/" >/dev/null 2>&1 && ts compile-cache-uploaded || ts compile-cache-upload-failed
) &
ts vllm-exec
/opt/runtime-venv/bin/vllm serve "$DEST" "$@" &
VLLM_PID=$!
# Kubernetes stops the container by sending SIGTERM to this script, which runs
# as process 1 and would not pass it on. Without this, vLLM keeps the node
# until the pod's grace period (an hour for this app) runs out.
trap 'kill -TERM "$VLLM_PID" 2>/dev/null' TERM INT
# A GPU fault can leave vLLM up but stuck: it accepts requests and never
# answers them. When requests are running and no tokens come out for
# STALL_SECONDS, stop vLLM so the container exits and restarts. The weights
# stay in the memory volume, so the restart skips the copy.
/opt/runtime-venv/bin/python -u -c "$WATCHDOG" "$VLLM_PID" "$SERVE_PORT" "${STALL_SECONDS:-300}" &
wait "$VLLM_PID"
"""

WATCHDOG = r"""
import os, signal, sys, time, urllib.request

pid, port, stall = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])


def counters():
    text = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=30).read().decode()
    running = tokens = 0.0
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            running += float(line.rsplit(" ", 1)[1])
        elif line.startswith(("vllm:generation_tokens_total", "vllm:prompt_tokens_total")):
            tokens += float(line.rsplit(" ", 1)[1])
    return running, tokens


last_tokens, last_progress = None, time.time()
while True:
    time.sleep(30)
    try:
        running, tokens = counters()
    except Exception:
        # Not up yet, or too busy to answer; only a stall with requests running counts.
        continue
    if tokens != last_tokens or running == 0:
        last_tokens, last_progress = tokens, time.time()
    elif time.time() - last_progress > stall:
        print(f"WATCHDOG no tokens for {stall:.0f}s with {running:.0f} requests running, stopping vLLM", flush=True)
        # The serve script then exits, and the container restarts with everything in it.
        os.kill(pid, signal.SIGKILL)
        sys.exit(1)
"""


def kimi_app(model: str, compile_remote: str, extra_args: list[str]) -> flyte.app.AppEnvironment:
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
            V1Volume(name="cache", empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit=CACHE_SIZE)),
        ],
        tolerations=[V1Toleration(key="amd.com/gpu", operator="Exists", effect="NoSchedule")],
    )
    vllm_args = [
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--tensor-parallel-size", str(GPUS_PER_NODE),
        "--trust-remote-code",
        "--reasoning-parser", "kimi_k3",
        "--kv-cache-dtype", "fp8",
        "--enable-prefix-caching",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92",
        # With more requests in flight, the image's kernels fail with
        # hipErrorLaunchOutOfResources on some batches (one had 188 requests).
        "--max-num-seqs", "64",
        # Prefixes evicted from GPU memory move to this much CPU memory (GiB), which
        # vLLM reads back instead of recomputing. It only helps when it is larger
        # than the GPU KV cache: 3.8M tokens, a bit over 500 GiB, on this node.
        "--kv-offloading-size", "700",
        *extra_args,
    ]
    return flyte.app.AppEnvironment(
        name=os.getenv("BATCH_APP_NAME", "kimi-k3"),
        image=image,
        command=["bash", "-c", SERVE, "serve", model, HF_REPO, CACHE_DIR, compile_remote, *vllm_args],
        port=PORT,
        # The node has about 2.7 TiB: the weights volume, the CPU KV cache and
        # what vLLM uses on the host.
        resources=flyte.Resources(cpu="48", memory="2500Gi", disk="200Gi"),
        # Not PORT: Knative reserves that name and refuses to create the service.
        env_vars={"VLLM_ROCM_USE_AITER": "1", "SERVE_PORT": str(PORT), "WATCHDOG": WATCHDOG},
        # vLLM checks requests against VLLM_API_KEY.
        secrets=flyte.Secret(key="kimi-vllm-api-key", as_env_var="VLLM_API_KEY"),
        scaling=flyte.app.Scaling(replicas=(1, 1)),
        requires_auth=False,
        cluster_pool=os.getenv("BATCH_CLUSTER_POOL"),
        timeouts=flyte.app.Timeouts(request=3600),
        # Startup includes waiting for the market to grant a node, so allow
        # far more than Knative's ten minute default.
        pod_template=flyte.PodTemplate(
            pod_spec=pod_spec,
            annotations={"serving.knative.dev/progress-deadline": "10800s"},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Remote path of the prefetched model directory")
    parser.add_argument("--extra-args", default="", help="Extra flags for vllm serve")
    parser.add_argument(
        "--compile-cache",
        default=None,
        help="Object store prefix for compiled kernels (default: next to the model)",
    )
    a = parser.parse_args()

    flyte.init_from_config()
    # Kernels depend on the image and the engine flags, so key the cache on both.
    import hashlib

    key = hashlib.sha256(f"{VLLM_BASE} {a.extra_args}".encode()).hexdigest()[:12]
    compile_remote = a.compile_cache or f"{a.model.rstrip('/')}-compile-cache/{key}"
    app = flyte.serve(kimi_app(a.model, compile_remote, a.extra_args.split()))
    print(app.url if hasattr(app, "url") else app)
