"""
Serve the base model and the RL-trained models with vLLM, plus a side-by-side Gradio chat.

    python serve.py                    # base + every trained model + chat UI
    python serve.py --tasks countdown  # only the models that passed review so far

Each model is its own single-GPU ``VLLMAppEnvironment`` (the worker pattern from flyte-sdk's
``examples/genai/vllm_lmcache/serve.py``), bound to a model **artifact** rather than a Hugging Face repo:

- ``base``         -> the ``Qwen2-5-1-5B-Instruct`` artifact that ``flyte.prefetch.hf_model`` published
- ``rl-countdown`` -> the ``qwen2-5-1-5b-rl-countdown`` artifact that ``train.py --task countdown`` published
- ``rl-json``      -> the ``qwen2-5-1-5b-rl-json`` artifact that ``train.py --task json`` published

What keeps it fast:

- ``stream_model=True``: safetensors stream from the object store straight into GPU memory; nothing is
  downloaded from Hugging Face or staged on local disk at startup.
- ``--enable-prefix-caching``: every request in a task shares the same long system prompt, so after the
  first request only the user's message is prefilled.
- Merged weights: the RL'd models are plain checkpoints, so there is no per-request LoRA overhead.
- Autoscaling: each model scales from zero to two replicas on demand and back to zero when idle, so a
  model nobody is using costs nothing. For a live talk, set ``MODEL_SCALING`` to ``(1, 1)`` first so
  nothing scales to zero mid-demo.

The LMCache L1/L2 tiers and prefix-aware router from that example are left out: they pay off across many
replicas under load, not for one presenter clicking through prompts.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from datetime import timedelta

import flyte
import flyte.app
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE, VLLMAppEnvironment

import chat_ui
import tasks
from train import BASE_ARTIFACT, MODEL_ARTIFACTS

INTERNAL_APP_ENDPOINT_PATTERN = "INTERNAL_APP_ENDPOINT_PATTERN"

# {{docs-fragment scaling}}
# Each model: no replicas while idle, a second replica when one is handling more than 16 requests at once,
# and back to zero 10 minutes after the last request.
MODEL_SCALING = flyte.app.Scaling(
    replicas=(0, 2),
    metric=flyte.app.Scaling.Concurrency(16),
    scaledown_after=timedelta(minutes=10),
)
# {{/docs-fragment scaling}}

# {{docs-fragment workers}}
base_worker = VLLMAppEnvironment(
    name="rl-demo-base",
    model_id="base",
    model_path=flyte.app.ArtifactValue(name=BASE_ARTIFACT, type="directory"),
    image=DEFAULT_VLLM_IMAGE,
    # bf16 needs compute capability >= 8.0; L4 is the cheapest GPU that qualifies, and a 1.5B model
    # leaves most of its 24GB for KV cache.
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="20Gi"),
    stream_model=True,
    extra_args=["--max-model-len", "4096", "--enable-prefix-caching", "--gpu-memory-utilization", "0.90"],
    scaling=MODEL_SCALING,
    requires_auth=True,
)

rl_workers = {
    task: base_worker.clone_with(
        name=f"rl-demo-rl-{task}",
        model_id=f"rl-{task}",
        model_path=flyte.app.ArtifactValue(name=MODEL_ARTIFACTS[task], type="directory"),
    )
    for task in tasks.TASKS
}
# {{/docs-fragment workers}}


# {{docs-fragment chat}}
chat_image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "gradio>=6.0",
    "openai>=1.40",
    "async-lru>=2.0.0",  # train.py is imported for the artifact names
    "flyteplugins-vllm",
    pre=True,
)

# No depends_on: the workers are served individually (only the trained ones), and the chat learns which
# models exist from the ``models`` parameter set at deploy time.
chat = flyte.app.AppEnvironment(
    name="rl-demo-chat",
    image=chat_image,
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    port=7860,
    parameters=[flyte.app.Parameter(name="models", value=",".join(["base", *(f"rl-{t}" for t in tasks.TASKS)]))],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=timedelta(minutes=10)),
    requires_auth=True,
)


@chat.server
def serve_chat(models: str) -> None:
    # Same in-cluster addressing as the vllm_lmcache router: workers are named rl-demo-<model_id>.
    pattern = os.environ[INTERNAL_APP_ENDPOINT_PATTERN]
    endpoints = {m: pattern.format(app_fqdn=f"rl-demo-{m}").rstrip("/") for m in models.split(",") if m}
    chat_ui.build_demo(endpoints).queue(default_concurrency_limit=8).launch(server_name="0.0.0.0", server_port=7860)


# {{/docs-fragment chat}}


# {{docs-fragment serve_entrypoint}}
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks", nargs="+", choices=tasks.TASKS, default=list(tasks.TASKS), help="trained models to serve"
    )
    args = parser.parse_args()

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent, log_level=logging.INFO)
    workers = [base_worker, *(rl_workers[t] for t in args.tasks)]
    for worker in workers:
        print(f"{worker.model_id}: {flyte.serve(worker).url}")
    models = ",".join(w.model_id for w in workers)
    app = flyte.with_servecontext(parameter_values={chat.name: {"models": models}}).serve(chat)
    print(f"chat: {app.url}")
# {{/docs-fragment serve_entrypoint}}
