# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "async-lru",
#    "torch",
#    "transformers",
# ]
# main = "embed"
# params = "model=s3://your-bucket/models/bge-base-en-v1.5 texts='[\"First document...\"]'"
# ///

"""Stream a transformers model's weights onto the GPU inside a task."""

# {{docs-fragment load}}
import flyte
from async_lru import alru_cache
from flyte.io import Dir

env = flyte.TaskEnvironment(
    name="streamed-embedder",
    image=flyte.Image.from_debian_base().with_pip_packages("async-lru", "torch", "transformers", "unionai-reuse"),
    resources=flyte.Resources(cpu=4, memory="8Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=8),
)


@alru_cache(maxsize=1)
async def get_model(model_path: str):
    """Load the model once per container."""
    import torch
    from flyte.extras.model_streamer import load_hf_model
    from transformers import AutoModel, AutoTokenizer

    model, local_dir = await load_hf_model(model_path, model_class=AutoModel, device="cuda", dtype=torch.float16)
    return model, AutoTokenizer.from_pretrained(local_dir)


@env.task
async def embed(model: Dir, texts: list[str]) -> list[list[float]]:
    import torch

    hf_model, tokenizer = await get_model(model.path)
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        cls = hf_model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1).float().cpu().tolist()


# {{/docs-fragment load}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        embed,
        model=Dir.from_existing_remote("s3://your-bucket/models/bge-base-en-v1.5"),
        texts=["First document..."],
    )
    print(run.url)
