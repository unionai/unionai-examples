# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# ///

"""Serve Kimi K3 with weights loaded by flyteplugins-vllm's streaming loader.

The same app as kimi_app.py, except for how the weights arrive. Instead of
copying them with s5cmd before vLLM starts, vLLM loads them through
`load_format="flyte-streaming"`. With `--node-cache-dir`, the first rank on
the node downloads the checkpoint into that directory once and every rank
reads it from there. Without it, every rank streams the whole checkpoint.

Startup prints a timestamp at each step, so the log shows how long each took.
On a 1.5 TB checkpoint the plugin's default reads are slow; 256 concurrent
reads of 64 MiB brought the node-cache load from 35 minutes to 6.

The released flyteplugins-vllm is installed into vLLM's virtualenv.
`--node-cache-dir` needs flyte-sdk#1871; until that is released, build the
flyte and flyteplugins-vllm wheels from it (`make dist dist-plugins`) and put
them in `_wheels/` next to this file, and the image installs those instead.

    uv run kimi_stream_app.py --model <model path> --node-cache-dir /cache/weights \
        --chunk-size 67108864 --max-concurrency 256
"""

import argparse
import hashlib
import os
from pathlib import Path

import flyte

from kimi_app import VLLM_BASE, kimi_app
from kimi_app import image as kimi_image

WHEELS = sorted((Path(__file__).parent / "_wheels").glob("*.whl"))
_base = kimi_image.clone(name="kimi-stream", extendable=True)
if WHEELS:
    _install = "/opt/wheels/*.whl"
    _base = _base.with_source_file(WHEELS, dst="/opt/wheels/")
else:
    _install = "flyteplugins-vllm"
image = _base.with_commands(
    [
        f"/opt/runtime-venv/bin/python -m pip install {_install}"
        f" || uv pip install --python /opt/runtime-venv/bin/python {_install}",
        "/opt/runtime-venv/bin/python -c 'import flyteplugins.vllm.model_streamer'",
    ]
)

# Arguments: model, HF repo (unused), cache dir, compile cache, node cache dir ("" for none),
# read size in bytes and concurrent reads ("" for the plugin's defaults), vllm args.
SERVE = r"""
set -euo pipefail
MODEL="$1"; CACHE="$3"; COMPILE_REMOTE="$4"; NODE_CACHE="$5"; CHUNK="$6"; CONCURRENCY="$7"; shift 7
ts() { echo "STARTUP $(date +%s) $*"; }
ts container-start
export AWS_ACCESS_KEY_ID="$FLYTE_AWS_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$FLYTE_AWS_SECRET_ACCESS_KEY" AWS_REGION=auto
COMPILE="$CACHE/compile"
mkdir -p "$COMPILE" "$CACHE/meta"
s5cmd --endpoint-url "$FLYTE_AWS_ENDPOINT" --numworkers 64 --log error cp "${COMPILE_REMOTE%/}/*" "$COMPILE/" >/dev/null 2>&1 || true
ts compile-cache-ready
export HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="$COMPILE/vllm" TORCHINDUCTOR_CACHE_DIR="$COMPILE/inductor" TRITON_CACHE_DIR="$COMPILE/triton"
/opt/runtime-venv/bin/python - "$MODEL" "$CACHE/meta" "$NODE_CACHE" "$CHUNK" "$CONCURRENCY" > /tmp/engine_args.json <<'EOF'
import asyncio, json, sys
from flyteplugins.vllm.model_streamer import engine_args
model, meta, node_cache, chunk, concurrency = sys.argv[1:6]
print(json.dumps(asyncio.run(engine_args(
    model, local_dir=meta, node_cache_dir=node_cache or None,
    chunk_size=int(chunk) if chunk else None, max_concurrency=int(concurrency) if concurrency else None,
))))
EOF
ts metadata-ready
LOCAL_MODEL=$(/opt/runtime-venv/bin/python -c "import json; print(json.load(open('/tmp/engine_args.json'))['model'])")
EXTRA=$(/opt/runtime-venv/bin/python -c "import json; print(json.dumps(json.load(open('/tmp/engine_args.json'))['model_loader_extra_config']))")
echo "ENGINE-ARGS model=$LOCAL_MODEL extra=$EXTRA"
(
  until /opt/runtime-venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:$SERVE_PORT/health', timeout=5)" >/dev/null 2>&1; do sleep 10; done
  ts server-ready
  [ -n "$NODE_CACHE" ] && du -sh "$NODE_CACHE" | sed 's/^/NODE-CACHE /'
) &
ts vllm-exec
exec /opt/runtime-venv/bin/vllm serve "$LOCAL_MODEL" --load-format flyte-streaming --model-loader-extra-config "$EXTRA" "$@"
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Remote path of the prefetched model directory")
    parser.add_argument("--node-cache-dir", default="", help="Download once per node into this directory")
    parser.add_argument("--chunk-size", default="", help="Bytes per ranged read (default: the plugin's)")
    parser.add_argument("--max-concurrency", default="", help="Concurrent ranged reads (default: the plugin's)")
    a = parser.parse_args()

    flyte.init_from_config()
    key = hashlib.sha256(f"{VLLM_BASE} ".encode()).hexdigest()[:12]
    compile_remote = f"{a.model.rstrip('/')}-compile-cache/{key}"
    base = kimi_app(a.model, compile_remote, [])
    # kimi_app's command is bash -c SERVE serve <model> <hf repo> <cache> <compile> <vllm args...>.
    _, _, _, name0, model, hf_repo, cache, compile_, *vllm_args = base.command
    # This app starts vLLM itself, so it has no watchdog to configure.
    env_vars = {k: v for k, v in (base.env_vars or {}).items() if k != "WATCHDOG"}
    app = base.clone_with(
        name=os.getenv("BATCH_APP_NAME", "kimi-k3-stream"),
        image=image,
        env_vars=env_vars,
        command=[
            "bash", "-c", SERVE, name0, model, hf_repo, cache, compile_,
            a.node_cache_dir, a.chunk_size, a.max_concurrency, *vllm_args,
        ],
    )
    print(flyte.serve(app).url)
