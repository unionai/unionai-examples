"""Container images for the legal-agent RL examples.

Every image starts from LAB's own sandbox image, so a task's pod has the tools LAB's agent expects (pandoc,
LibreOffice, poppler, ripgrep, the ``parse-doc`` parsers, and the docx/pptx libraries) at the versions LAB pins.
The GPU images add vLLM and the training stack on top; ``ROCM_IMAGE`` for AMD Instinct GPUs (MI355X) and
``CUDA_IMAGE`` for NVIDIA GPUs. Neither inherits a ``vllm serve`` entrypoint, because neither starts from a vLLM image.
"""

import flyte

# LAB's sandbox image at the digest that `latest` pointed to on 2026-11-25 (harvey-labs v1.4.0, commit 7f9b1ee).
LAB_SANDBOX = "ghcr.io/harveyai/lab-sandbox@sha256:13aaf4d0ba3fd0436a4f6a0793b3a35473bcc539bcbc079c977b47539e6ba1af"
LAB_COMMIT = "7f9b1ee19f9224576a5a5143c4e6f75842ac7020"
# lab_core 1.4.0 has no published wheel (the latest release is 1.3.0), so install it from the pinned commit with a
# sparse clone that skips the 3 GB tasks/ tree. Tasks are fetched per run, at the same commit (lab_harness.fetch_lab).
_INSTALL_LAB_CORE = (
    "git clone -q --filter=blob:none --no-checkout https://github.com/harveyai/harvey-labs.git /tmp/hl && cd /tmp/hl"
    " && git sparse-checkout set --no-cone /lab_core/ /pyproject.toml /README.md /LICENSE"
    f" && git checkout -q {LAB_COMMIT} && uv pip install /tmp/hl && cd / && rm -rf /tmp/hl"
)

VLLM_VERSION = "0.34.0"
FLYTE_VERSION = "2.13.0"
_ROCM_WHEELS = "https://wheels.vllm.ai/rocm"
_COMMON = ("transformers>=5.20", "peft>=0.22", "accelerate>=1.10", "pandas", "pyarrow", "numpy", "scipy",
           "openai>=1.100", "httpx", "kubernetes", "pydantic>=2.10", "unionai-reuse>=0.1.11")


def _lab_base(name: str) -> flyte.Image:
    # The sandbox image is python:3.12-slim underneath. Install everything, Flyte included, into that interpreter
    # (not a separate virtualenv), so the task entrypoint is on PATH and the agent's `python3` sees the same packages.
    return (flyte.Image.from_base(LAB_SANDBOX).clone(name=name, python_version=(3, 12), extendable=True)
            .with_env_vars({"UV_PYTHON": "/usr/local/bin/python3", "UV_BREAK_SYSTEM_PACKAGES": "1",
                            "UV_SYSTEM_PYTHON": "1"})
            .with_pip_packages(f"flyte=={FLYTE_VERSION}"))


# {{docs-fragment images}}
# vLLM publishes ROCm wheels (with a matching torch and the ROCm SDK as pip packages) at wheels.vllm.ai/rocm.
ROCM_IMAGE = (
    _lab_base("legal-agent-rl-rocm")
    # flash-attn's ROCm build is a wheel on that index; pin it (without deps) so the resolver can't pick PyPI's
    # source tarball, which tries to compile against CUDA.
    .with_pip_packages(f"flash-attn @ {_ROCM_WHEELS}/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/rocm100-wheels/"
                       "flash_attn-2.8.3-cp312-cp312-manylinux_2_34_x86_64.whl", extra_args="--no-deps")
    .with_pip_packages(f"vllm=={VLLM_VERSION}+rocm100", index_url=f"{_ROCM_WHEELS}/",
                       extra_index_urls=["https://pypi.org/simple"], extra_args="--index-strategy unsafe-best-match")
    .with_pip_packages(*_COMMON)
    .with_commands([_INSTALL_LAB_CORE])
    .with_env_vars({"HSA_NO_SCRATCH_RECLAIM": "1", "TOKENIZERS_PARALLELISM": "false"})
)
CUDA_IMAGE = (
    _lab_base("legal-agent-rl-cuda")
    .with_pip_packages(f"vllm=={VLLM_VERSION}", *_COMMON)
    .with_commands([_INSTALL_LAB_CORE])
    .with_env_vars({"TOKENIZERS_PARALLELISM": "false"})
)
# {{/docs-fragment images}}
# Judging with hosted models and analysis: no GPU stack, but the same parsers LAB grades with.
CPU_IMAGE = (_lab_base("legal-agent-rl-cpu").with_pip_packages(*_COMMON, "anthropic")
             .with_commands([_INSTALL_LAB_CORE]))
