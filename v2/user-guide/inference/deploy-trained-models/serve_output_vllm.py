# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "flyteplugins-vllm>=2.12.2",
# ]
# ///

"""Serve a task output with vLLM."""

# {{docs-fragment app}}
import flyte
import flyte.app
from flyte.remote import App
from flyteplugins.vllm import VLLMAppEnvironment

app_env = VLLMAppEnvironment(
    name="support-llm",
    model_path=flyte.app.RunOutput(type="directory", task_name="training.finetune"),
    model_id="support-llm",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1", disk="50Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=600),
)

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.deploy(app_env)
    App.get(name=app_env.name).activate()
# {{/docs-fragment app}}
