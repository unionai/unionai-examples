"""Shared Flyte environments and climbmix dataset bundle tasks."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import flyte
from flyte.io import Dir

from autoresearch_types import DatasetProfile
from autoresearch_types import DEFAULT_NUM_SHARDS

TRAIN_PIP_PACKAGES = ["torch", "numpy", "pyarrow", "requests", "tiktoken", "rustbpe"]

image = flyte.Image.from_debian_base(name="mle-autoresearch").with_pip_packages(
    "litellm",
    "httpx",
    "pydantic-monty",
    "unionai-sandbox[flyte]",
    *TRAIN_PIP_PACKAGES,
)

bundle_env = flyte.TaskEnvironment(
    name="autoresearch-bundle",
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    image=image,
)

experiment_env = flyte.TaskEnvironment(
    name="autoresearch-experiment",
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    image=image,
)

# {{docs-fragment env}}
agent_env = flyte.TaskEnvironment(
    name="autoresearch-agent",
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    image=image,
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    depends_on=[experiment_env, bundle_env],
)
# {{/docs-fragment env}}


@dataclass
class AutoresearchBundle:
    data_dir: Dir
    tokenizer_dir: Dir


@bundle_env.task(cache="auto")
async def build_bundle(num_shards: int = DEFAULT_NUM_SHARDS, download_workers: int = 4) -> AutoresearchBundle:
    """Download climbmix shards + train the BPE tokenizer; cache the result."""
    import prepare

    cache = tempfile.mkdtemp(prefix="autoresearch-cache-")
    os.environ["AUTORESEARCH_CACHE"] = cache
    prepare.download_data(num_shards, download_workers=download_workers)
    prepare.train_tokenizer()
    data_dir = await Dir.from_local(prepare.data_dir())
    tokenizer_dir = await Dir.from_local(prepare.tokenizer_dir())
    return AutoresearchBundle(data_dir=data_dir, tokenizer_dir=tokenizer_dir)


@bundle_env.task(cache="auto")
async def profile_bundle(bundle: AutoresearchBundle) -> DatasetProfile:
    """Summarize the prepared bundle for the agent's context."""
    import prepare

    data_dir = await bundle.data_dir.download()
    tokenizer_dir = await bundle.tokenizer_dir.download()
    parquet_files = sorted(p.name for p in Path(data_dir).glob("*.parquet"))
    data_bytes = sum(p.stat().st_size for p in Path(data_dir).glob("**/*") if p.is_file())
    tok_bytes = sum(p.stat().st_size for p in Path(tokenizer_dir).glob("**/*") if p.is_file())
    return DatasetProfile(
        n_parquet_files=len(parquet_files),
        parquet_files=parquet_files,
        vocab_size=prepare.VOCAB_SIZE,
        data_bytes=data_bytes,
        tokenizer_bytes=tok_bytes,
    )


async def materialize_cache(bundle: AutoresearchBundle) -> str:
    """Download the bundle into an AUTORESEARCH_CACHE-shaped scratch dir."""
    cache = tempfile.mkdtemp(prefix="autoresearch-run-")
    os.environ["AUTORESEARCH_CACHE"] = cache
    await bundle.data_dir.download(local_path=os.path.join(cache, "data"))
    await bundle.tokenizer_dir.download(local_path=os.path.join(cache, "tokenizer"))
    return cache
