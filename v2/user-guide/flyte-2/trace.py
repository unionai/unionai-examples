# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "World"
# ///

# {{docs-fragment all}}
# https://github.com/unionai/unionai-examples/blob/main/v2/user-guide/flyte-2/async.py

import flyte

env = flyte.TaskEnvironment(name="trace_example_env")

@flyte.trace
async def call_llm(prompt: str) -> str:
    return "Initial response from LLM"

@env.task
def finalize_output(output: str) -> str:
    return "Finalized output"

@env.task(cache=flyte.Cache(behavior="auto"))
async def main(prompt: str) -> str:
    output = await call_llm(prompt)
    output = await finalize_output(output)
    return output

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main, prompt="Prompt to LLM")
    print(run.name)
    print(run.url)
    run.wait()
# {{docs-fragment all}}