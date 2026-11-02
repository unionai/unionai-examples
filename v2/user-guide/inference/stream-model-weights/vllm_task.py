# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "flyteplugins-vllm>=2.12.2",
#    "async-lru",
# ]
# main = "generate"
# params = "model=s3://your-bucket/models/smollm2 prompts='[\"What is object storage?\"]'"
# ///

"""Stream a vLLM engine's weights from object storage inside a task."""

# {{docs-fragment vllm}}
import asyncio

import flyte
from async_lru import alru_cache
from flyte.io import Dir
from flyteplugins.vllm import DEFAULT_VLLM_IMAGE

env = flyte.TaskEnvironment(
    name="streamed-vllm",
    image=DEFAULT_VLLM_IMAGE.clone(name="streamed-vllm").with_pip_packages("async-lru", "unionai-reuse"),
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="L4:1"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=4),
)


@alru_cache(maxsize=1)
async def get_llm(model_path: str):
    """Build the engine once per container."""
    from flyteplugins.vllm.model_streamer import engine_args
    from vllm import LLM

    args = await engine_args(model_path)
    # LLM() blocks for the whole load, so run it off the event loop.
    return await asyncio.to_thread(LLM, **args, max_model_len=4096)


# Concurrent calls on a replica share the engine, and the offline LLM isn't re-entrant.
generate_lock = asyncio.Lock()


@env.task
async def generate(model: Dir, prompts: list[str]) -> list[str]:
    from vllm import SamplingParams

    llm = await get_llm(model.path)
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    async with generate_lock:
        outputs = await asyncio.to_thread(llm.chat, messages, SamplingParams(max_tokens=128))
    return [output.outputs[0].text for output in outputs]


# {{/docs-fragment vllm}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        generate,
        model=Dir.from_existing_remote("s3://your-bucket/models/smollm2"),
        prompts=["What is object storage?"],
    )
    print(run.url)
