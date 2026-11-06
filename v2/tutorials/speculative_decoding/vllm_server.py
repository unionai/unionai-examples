"""Run a vLLM OpenAI-compatible server as a subprocess of a GPU task.

Data generation, training, and benchmarking all talk to vLLM over HTTP, the same way
production traffic does. This module starts the server, waits until it is healthy, and
reads the speculative decoding counters from its Prometheus endpoint.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

import httpx

from config import TARGET_MODEL

# Qwen3 emits tool calls in the Hermes format.
TOOL_ARGS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]


@dataclass
class VLLMServer:
    """A running `vllm serve` process."""

    port: int
    process: subprocess.Popen
    log_path: str

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()


def start_server(
    *,
    model: str = TARGET_MODEL,
    port: int = 8000,
    gpus: str | None = None,
    speculative_config: dict | None = None,
    extra_args: list[str] | None = None,
    hidden_states_path: str | None = None,
    timeout_s: int = 1800,
) -> VLLMServer:
    """Start vLLM and block until `/health` answers.

    With `hidden_states_path`, the server also writes the target's hidden states for every
    request, which is how the trainer gets them in online mode.
    """
    env = dict(os.environ)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    extra_args = list(extra_args or [])

    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port)]
    if hidden_states_path is not None:
        cmd += hidden_state_args(model, hidden_states_path)
    elif speculative_config is not None:
        cmd += ["--speculative-config", json.dumps(speculative_config)]
    cmd += extra_args

    log_path = f"/tmp/vllm-{port}.log"
    log = open(log_path, "w")
    print("Starting:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    server = VLLMServer(port=port, process=process, log_path=log_path)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}:\n{tail(log_path)}")
        try:
            if httpx.get(f"{server.base_url}/health", timeout=5).status_code == 200:
                print(f"vLLM ready on port {port}", flush=True)
                return server
        except httpx.HTTPError:
            pass
        time.sleep(5)
    server.stop()
    raise TimeoutError(f"vLLM did not become healthy in {timeout_s}s:\n{tail(log_path)}")


# {{docs-fragment hidden-states}}
def hidden_state_args(model: str, path: str) -> list[str]:
    """vLLM flags that write the target's hidden states to `path` for every request.

    EAGLE-3 fuses three of the target's layers (early, middle, late); the final layer is
    captured too, for the training targets. These are speculators' default layer choices.
    """
    from transformers import AutoConfig

    n = AutoConfig.from_pretrained(model).num_hidden_layers
    layers = [2, n // 2, n - 3, n]
    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layers}},
    }
    kv_transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": path},
    }
    return [
        "--speculative-config", json.dumps(speculative_config),
        "--kv-transfer-config", json.dumps(kv_transfer_config),
        # Enables /v1/chat/completions/render, which prepare-data uses to tokenize
        # raw conversations with the target's chat template.
        "--enable-scale-out",
    ]
# {{/docs-fragment hidden-states}}


def tail(path: str, lines: int = 60) -> str:
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-lines:])


# {{docs-fragment metrics}}
@dataclass
class SpecDecodeCounters:
    """Cumulative speculative decoding counters from vLLM's `/metrics` endpoint."""

    drafts: float = 0.0
    draft_tokens: float = 0.0
    accepted_tokens: float = 0.0
    accepted_per_pos: dict[int, float] = field(default_factory=dict)

    def __sub__(self, other: "SpecDecodeCounters") -> "SpecDecodeCounters":
        return SpecDecodeCounters(
            drafts=self.drafts - other.drafts,
            draft_tokens=self.draft_tokens - other.draft_tokens,
            accepted_tokens=self.accepted_tokens - other.accepted_tokens,
            accepted_per_pos={
                k: v - other.accepted_per_pos.get(k, 0.0) for k, v in self.accepted_per_pos.items()
            },
        )

    @property
    def acceptance_length(self) -> float:
        """Mean tokens produced per verification step: 1 + accepted / drafts."""
        return 1.0 + self.accepted_tokens / self.drafts if self.drafts else 1.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens the target accepted."""
        return self.accepted_tokens / self.draft_tokens if self.draft_tokens else 0.0


_METRIC = re.compile(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$')


def read_counters(base_url: str) -> SpecDecodeCounters:
    """Scrape vLLM's Prometheus endpoint, summing across label sets."""
    counters = SpecDecodeCounters()
    for line in httpx.get(f"{base_url}/metrics", timeout=10).text.splitlines():
        match = _METRIC.match(line)
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        if name == "vllm:spec_decode_num_drafts":
            counters.drafts += value
        elif name == "vllm:spec_decode_num_draft_tokens":
            counters.draft_tokens += value
        elif name == "vllm:spec_decode_num_accepted_tokens":
            counters.accepted_tokens += value
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                k = int(pos.group(1))
                counters.accepted_per_pos[k] = counters.accepted_per_pos.get(k, 0.0) + value
    return counters
# {{/docs-fragment metrics}}
