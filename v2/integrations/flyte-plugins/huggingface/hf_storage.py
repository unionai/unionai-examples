# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "huggingface_hub",
# ]
# main = "pipeline"
# params = "name=HuggingFace"
# ///
"""Store offloaded File and Dir bytes in a Hugging Face storage bucket.

This example does not use the Hugging Face plugin. It points `raw_data_path` at
an `hf://` prefix, so the bytes behind `flyte.io.File` and `flyte.io.Dir` land in
a Hugging Face storage bucket instead of the deployment's default object store.

Set HF_BUCKET to `<username>/<bucket>` and HF_TOKEN to a token with write access
to it.
"""

import os
import tempfile

# {{docs-fragment env}}
import flyte
from flyte.io import File

# huggingface_hub only. The Hugging Face plugin is not involved in raw data storage.
env = flyte.TaskEnvironment(
    name="hf_storage_env",
    image=flyte.Image.from_debian_base(name="hf_storage").with_pip_packages("huggingface_hub"),
    secrets=[flyte.Secret(key="huggingface-token", as_env_var="HF_TOKEN")],
)
# {{/docs-fragment env}}


# {{docs-fragment tasks}}
@env.task
async def write_greeting(name: str) -> File:
    """Write a file. Its bytes land under raw_data_path, wherever that points."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"Hello, {name}!")
        tmp_path = f.name
    # Prints the hf:// prefix on a run configured for one.
    print(flyte.ctx().raw_data_path)
    return await File.from_local(tmp_path)


@env.task
async def read_and_transform(file: File) -> str:
    """Read the file back. The download is the same call regardless of backend."""
    local_path = await file.download()
    with open(local_path, "r") as f:
        return f.read().upper()


@env.task
async def pipeline(name: str) -> str:
    file = await write_greeting(name)
    return await read_and_transform(file)
# {{/docs-fragment tasks}}


# {{docs-fragment main}}
if __name__ == "__main__":
    # "<username>/<bucket>", naming a Hugging Face storage bucket you can write to.
    HF_BUCKET = os.environ["HF_BUCKET"]

    flyte.init_from_config()

    run = flyte.with_runcontext(
        raw_data_path=f"hf://buckets/{HF_BUCKET}/raw-data/",
    ).run(pipeline, name="HuggingFace")

    print(run.url)
# {{/docs-fragment main}}
