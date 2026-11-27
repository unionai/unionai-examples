"""vLLM servers inside a task's pod: one per model, pinned to a set of GPUs, started once per container.

On a whole 8-GPU node the policy, the judge and the trainer share the machine, so each server gets an explicit GPU
set (``HIP_VISIBLE_DEVICES`` on AMD, ``CUDA_VISIBLE_DEVICES`` on NVIDIA). Servers outlive a single task when the
environment is reusable, which is what keeps a 65 GB judge loaded across the cells of an eval matrix.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx
from pydantic import BaseModel

IS_ROCM = os.path.exists("/dev/kfd")
VISIBLE = "HIP_VISIBLE_DEVICES" if IS_ROCM else "CUDA_VISIBLE_DEVICES"


class ServerSpec(BaseModel):
    """One OpenAI-compatible vLLM server."""

    served_name: str
    hf_id: str
    gpus: list[int]
    tensor_parallel: int = 1  # data parallel = len(gpus) // tensor_parallel
    max_model_len: int = 65536
    gpu_memory_utilization: float = 0.85
    enable_lora: bool = False
    max_lora_rank: int = 64
    max_loras: int = 2  # adapters resident at once (a checkpoint matrix keeps them all)
    tool_parser: str = ""  # vLLM's name for the model's tool-call format; "" picks one from the model id
    reasoning_parser: str = ""
    extra_args: list[str] = []
    env: dict[str, str] = {}  # per-server environment overrides
    port: int = 0  # 0: pick from the first GPU

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.resolved_port}/v1"

    @property
    def resolved_port(self) -> int:
        return self.port or 8100 + min(self.gpus)


_procs: dict[str, subprocess.Popen] = {}
_lock = asyncio.Lock()


# vLLM 0.34's parsers for the models used here (each model card's recommended serve flags).
PARSERS = {"gpt-oss": ("openai", "openai_gptoss"), "nemotron-3-nano": ("qwen3_coder", "nemotron_v3"),
           "olmo-3": ("olmo3", "")}


def parsers_for(hf_id: str) -> tuple[str, str]:
    return next((v for k, v in PARSERS.items() if k in hf_id.lower()), ("hermes", ""))


def _cmd(s: ServerSpec) -> list[str]:
    tool, reasoning = parsers_for(s.hf_id)
    tool, reasoning = s.tool_parser or tool, s.reasoning_parser or reasoning
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", s.hf_id,
           "--served-model-name", s.served_name, "--port", str(s.resolved_port), "--host", "127.0.0.1",
           "--tensor-parallel-size", str(s.tensor_parallel),
           "--data-parallel-size", str(max(1, len(s.gpus) // s.tensor_parallel)),
           "--max-model-len", str(s.max_model_len), "--gpu-memory-utilization", str(s.gpu_memory_utilization),
           "--enable-prefix-caching", "--enable-auto-tool-choice", "--tool-call-parser", tool, "--trust-remote-code",
           *(["--reasoning-parser", reasoning] if reasoning else []), *s.extra_args]
    if s.enable_lora:
        cmd += ["--enable-lora", "--max-lora-rank", str(s.max_lora_rank), "--max-loras", str(s.max_loras)]
    return cmd


async def ensure(s: ServerSpec, timeout_s: int = 3600) -> float:
    """Start the server if it isn't running in this container; return seconds spent waiting for it."""
    async with _lock:
        p = _procs.get(s.served_name)
        if p is not None and p.poll() is None:
            return 0.0
        # aiter's kernels compile at first use and need the ROCm compiler, which the pip-installed ROCm stack doesn't
        # ship: with aiter on, the engine dies on the first request (a failed `ninja` build). vLLM's other ROCm
        # kernels need no compiler.
        rocm = {"VLLM_ROCM_USE_AITER": "0"} if IS_ROCM else {}
        env = {**os.environ, **rocm, VISIBLE: ",".join(map(str, s.gpus)), "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
               "HF_HUB_ENABLE_HF_TRANSFER": "0", **s.env}
        print("starting vLLM:", " ".join(_cmd(s)), f"({VISIBLE}={env[VISIBLE]})", flush=True)
        log = open(f"/tmp/vllm-{s.served_name}.log", "w")
        _procs[s.served_name] = subprocess.Popen(_cmd(s), env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - t0 < timeout_s:
            if _procs[s.served_name].poll() is not None:
                tail = open(f"/tmp/vllm-{s.served_name}.log").read()[-3000:]
                print(tail, flush=True)
                raise RuntimeError(f"vLLM server {s.served_name} exited with {_procs[s.served_name].returncode}: "
                                   f"{tail[-1200:]}")
            try:
                if (await client.get(s.url.replace("/v1", "/health"), timeout=5)).status_code == 200:
                    return time.time() - t0
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError(f"vLLM server {s.served_name} not healthy after {timeout_s}s")


async def load_lora(s: ServerSpec, name: str, path: str) -> None:
    """Hot-load an adapter into a running server; rollouts then ask for ``model=name``."""
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(s.url.replace("/v1", "/v1/load_lora_adapter"),
                              json={"lora_name": name, "lora_path": path})
        if r.status_code != 200 and "already" not in r.text:
            raise RuntimeError(f"load_lora_adapter({name}) -> {r.status_code}: {r.text[:300]}")


async def unload_lora(s: ServerSpec, name: str) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(s.url.replace("/v1", "/v1/unload_lora_adapter"), json={"lora_name": name})


def stop(name: str | None = None) -> None:
    for n in [name] if name else list(_procs):
        p = _procs.pop(n, None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()


def log_errors(name: str, n: int = 30) -> list[str]:
    """The first error-looking lines of a server's log: the root cause, not the shutdown that follows it."""
    try:
        lines = open(f"/tmp/vllm-{name}.log").read().splitlines()
    except OSError:
        return []
    return [l[:300] for l in lines if any(k in l for k in ("Error", "error:", "Traceback", "FAILED", "failed"))][:n]


def gpu_report() -> list[dict]:
    """Per-GPU memory, read from the kernel so it works the same on AMD and NVIDIA."""
    import glob

    rows = []
    for card in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        d = os.path.dirname(card)
        try:
            total, used = (int(open(os.path.join(d, f)).read()) for f in ("mem_info_vram_total", "mem_info_vram_used"))
            rows.append({"card": d.split("/")[-2], "used_gib": round(used / 2**30, 1), "total_gib": round(total / 2**30)})
        except OSError:
            pass
    return rows
