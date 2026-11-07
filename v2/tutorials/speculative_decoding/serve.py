# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte==2.13.0",
#    "flyteplugins-vllm==2.13.0",
#    "httpx>=0.27",
# ]
# ///
"""Serve Qwen3-8B with a trained draft model, and run an agent against it.

- `drafter_app(method)` builds a vLLM app for any draft model family: it loads the target
  model from Hugging Face and the drafter from the latest run of that family's pipeline, and
  enables speculative decoding with the family's draft length.
- `agent_job` is the AI application: a tool-calling agent that calls one of the apps. Every
  model call is a `flyte.trace`, and the job returns its episodes as a trace file, which the
  retraining loop in `flywheel.py` turns into training data.

    python serve.py                          # serve the EAGLE-3 drafter
    python serve.py --method dflash          # serve the DFlash (or dspark) drafter
    python serve.py --run <name>             # pin the drafter from a specific run
    python serve.py --agent 200              # send 200 agent episodes through the app
"""

import asyncio
import json
import sys
import time
from functools import partial

import flyte
import flyte.app
from flyte.io import File
from flyteplugins.vllm import VLLMAppEnvironment

import data_reports
from config import ONE_GPU, TARGET_MODEL, cpu_image, gpu_image
from recipes import RECIPES
from vllm_server import TOOL_ARGS
from workloads import http_complete, load_agent_tasks, run_agent

API_KEY = flyte.Secret(key="specdec-api-key", as_env_var="VLLM_API_KEY")

# {{docs-fragment app}}
# The pipeline task whose latest output each family's app serves.
PIPELINES = {
    "eagle3": "specdec-driver.main",
    "dflash": "specdec-driver.main_dflash",
    "dspark": "specdec-driver.main_dspark",
}


def drafter_app(method: str) -> VLLMAppEnvironment:
    """A vLLM app serving Qwen3-8B with the latest drafter of one family."""
    return VLLMAppEnvironment(
        name=f"qwen3-8b-{method}",
        image=gpu_image,
        model_hf_path=TARGET_MODEL,
        model_id=TARGET_MODEL,
        draft_model_path=flyte.app.RunOutput(type="directory", task_name=PIPELINES[method]),
        speculative_config={
            "method": method,
            "num_speculative_tokens": RECIPES[method].num_speculative_tokens,
        },
        resources=ONE_GPU,
        # Scale to zero after 30 idle minutes: an idle app holds no GPU.
        scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
        requires_auth=False,
        secrets=API_KEY,
        extra_args=[*TOOL_ARGS, "--max-model-len", "8192", "--api-key", "$VLLM_API_KEY"],
    )


APPS = {method: drafter_app(method) for method in PIPELINES}
llm_app = APPS["eagle3"]
# {{/docs-fragment app}}

# {{docs-fragment agent-env}}
agent_env = flyte.TaskEnvironment(
    name="specdec-agent",
    image=cpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=API_KEY,
    # Deploys every app with the agent. Each scales to zero until it gets traffic.
    depends_on=list(APPS.values()),
)
# {{/docs-fragment agent-env}}


# {{docs-fragment agent-job}}
@flyte.trace
async def call_model(endpoint: str, messages: list[dict], tools: list[dict]) -> dict:
    """One model call. As a trace, its inputs and output are recorded and visible in the UI."""
    import os

    import httpx

    async with httpx.AsyncClient() as client:
        complete = http_complete(client, endpoint, api_key=os.environ["VLLM_API_KEY"])
        return await complete(messages, tools)


@agent_env.task(report=True)
async def agent_job(
    num_episodes: int = 100, method: str = "eagle3", seed: int = 0, concurrency: int = 16
) -> File:
    """Serve agent requests through one family's app and return the episodes as traces."""
    endpoint = APPS[method].endpoint
    tasks = load_agent_tasks("traffic", num_episodes, seed=seed)
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []

    async def one(task: dict) -> dict | None:
        async with semaphore:
            start = time.perf_counter()
            try:
                trace = await run_agent(partial(call_model, endpoint), task)
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
        data_reports.traffic_report(traces, len(tasks) - len(traces), latencies, title=f"Agent traffic through the {method} app")
    )
    await flyte.report.flush.aio()
    return await File.from_local(path)
# {{/docs-fragment agent-job}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    method = sys.argv[sys.argv.index("--method") + 1] if "--method" in sys.argv else "eagle3"
    app = APPS[method]
    if "--run" in sys.argv:
        run_name = sys.argv[sys.argv.index("--run") + 1]
        app = app.clone_with(
            name=app.name,
            draft_model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
        )
    if "--agent" in sys.argv:
        n = int(sys.argv[sys.argv.index("--agent") + 1])
        run = flyte.run(agent_job, num_episodes=n, method=method, seed=n)
        print(run.url)
    else:
        # Loading two models and capturing CUDA graphs takes a while on first start.
        print(flyte.with_servecontext(activate_timeout=1800.0).serve(app).url)
# {{/docs-fragment deploy}}
