# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "prompt='Prompt to LLM'"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="trace_example_env")

@flyte.trace
async def call_llm(prompt: str) -> str:
    return "Initial response from LLM"

@env.task
async def finalize_output(output: str) -> str:
    return "Finalized output"

@env.task(cache=flyte.Cache(behavior="auto"))
async def main(prompt: str) -> str:
    output = await call_llm(prompt)
    output = await finalize_output(output)
    return output

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, prompt="Prompt to LLM")
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}