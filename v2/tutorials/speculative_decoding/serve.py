# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# ///
"""Serve Qwen3-8B with the trained draft head, and run an agent against it.

- `llm_app` is a vLLM app that loads the target model from Hugging Face and the draft head
  from a pipeline run's output, and enables EAGLE-3 speculative decoding.
- `agent_job` is the AI application: a tool-calling agent that calls the app. Every model
  call is a `flyte.trace`, and the job returns its episodes as a trace file, which the
  retraining loop in `flywheel.py` turns into training data.

    python serve.py                 # deploy the app with the latest pipeline run's drafter
    python serve.py --run <name>    # pin the drafter from a specific run
    python serve.py --agent 200     # then send 200 agent episodes through it
"""

import asyncio
import json
import sys
import time

import flyte
import flyte.app
from flyte.io import File
from flyteplugins.vllm import VLLMAppEnvironment

import data_reports
from config import NUM_SPECULATIVE_TOKENS, ONE_GPU, TARGET_MODEL, cpu_image, gpu_image
from vllm_server import TOOL_ARGS
from workloads import http_complete, load_agent_tasks, run_agent

API_KEY = flyte.Secret(key="specdec-api-key", as_env_var="VLLM_API_KEY")

# {{docs-fragment app}}
llm_app = VLLMAppEnvironment(
    name="qwen3-8b-specdec",
    image=gpu_image,
    model_hf_path=TARGET_MODEL,
    model_id=TARGET_MODEL,
    # The draft head is the Dir returned by the latest successful `main` run.
    draft_model_path=flyte.app.RunOutput(type="directory", task_name="specdec-driver.main"),
    speculative_config={"method": "eagle3", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS},
    resources=ONE_GPU,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
    requires_auth=False,
    secrets=API_KEY,
    extra_args=[*TOOL_ARGS, "--max-model-len", "8192", "--api-key", "$VLLM_API_KEY"],
)
# {{/docs-fragment app}}

# {{docs-fragment agent-env}}
agent_env = flyte.TaskEnvironment(
    name="specdec-agent",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=API_KEY,
    depends_on=[llm_app],
)
# {{/docs-fragment agent-env}}


# {{docs-fragment agent-job}}
@flyte.trace
async def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """One model call. As a trace, its inputs and output are recorded and visible in the UI."""
    import os

    import httpx

    async with httpx.AsyncClient() as client:
        complete = http_complete(client, llm_app.endpoint, api_key=os.environ["VLLM_API_KEY"])
        return await complete(messages, tools)


@agent_env.task(report=True)
async def agent_job(num_episodes: int = 100, seed: int = 0, concurrency: int = 16) -> File:
    """Serve agent requests through the app and return the episodes as traces."""
    tasks = load_agent_tasks("traffic", num_episodes, seed=seed)
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(task: dict) -> dict | None:
        async with semaphore:
            start = time.perf_counter()
            try:
                trace = await run_agent(call_model, task)
            except Exception as e:
                print(f"episode failed: {e}")
                return None
            latencies.append(time.perf_counter() - start)
            return trace

    traces = [t for t in await asyncio.gather(*(one(t) for t in tasks)) if t]
    path = "/tmp/traces.jsonl"
    with open(path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    await flyte.report.replace.aio(
        data_reports.traffic_report(traces, len(tasks) - len(traces), latencies)
    )
    await flyte.report.flush.aio()
    return await File.from_local(path)
# {{/docs-fragment agent-job}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = llm_app
    if "--run" in sys.argv:
        run_name = sys.argv[sys.argv.index("--run") + 1]
        app = llm_app.clone_with(
            name=llm_app.name,
            draft_model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
        )
    if "--agent" in sys.argv:
        n = int(sys.argv[sys.argv.index("--agent") + 1])
        run = flyte.run(agent_job, num_episodes=n, seed=n)
        print(run.url)
    else:
        # Loading two models and capturing CUDA graphs takes a while on first start.
        print(flyte.with_servecontext(activate_timeout=1800.0).serve(app).url)
# {{/docs-fragment deploy}}
