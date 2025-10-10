@flyte.trace
async def call_llm(prompt: str) -> str:
    return ...

@env.task
def finalize_output(output: str) -> str:
    return ...

@env.task(cache=flyte.Cache(behavior="auto"))
async def main(prompt: str) -> str:
    output = await call_llm(prompt)
    output = await finalize_output(output)
    return output