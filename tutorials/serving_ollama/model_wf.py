from pathlib import Path
import subprocess
from typing import Annotated

from union import task, workflow, ImageSpec
from flytekit import Artifact
from flytekit.types.directory import FlyteDirectory
from flytekit.core.artifact import Artifact, Inputs
import flytekit

import typing

MyOllamaModel = Artifact(name="ollama_model", partition_keys=["model"])

# We will use the `ImageSpec` class to define the container image for our application.
# We will use the `unionai` builder to build the image. The packages required for
# our application are `union-runtime` and `ollama`. The `union-runtime` package
# provides the necessary runtime environment for our application, while the `ollama`
# package provides the necessary tools to run the Ollama model.
image = ImageSpec(
    builder="union",
    name="ollama-serve",
    apt_packages=["curl", "systemctl"],
    commands=[  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "rm -f ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama ollama",
        "mkdir -p /usr/share/ollama/.ollama/models",
    ],
    packages=["union", "ollama", "httpx", "loguru"],
)


def wait_for_ollama(timeout: int = 30, interval: int = 2) -> None:
    """Wait for Ollama service to be ready.

    :param timeout: Maximum time to wait in seconds
    :param interval: Time between checks in seconds
    """
    import httpx
    from loguru import logger
    import time

    start_time = time.time()
    while True:
        try:
            response = httpx.get("http://localhost:11434/api/version")
            if response.status_code == 200:
                logger.info("Ollama service is ready")
                return
        except httpx.ConnectError:
            if time.time() - start_time > timeout:
                raise TimeoutError("Ollama service failed to start")
            logger.info(
                f"Waiting for Ollama service... ({int(time.time() - start_time)}s)"
            )
            time.sleep(interval)


# Use cache=True if we will use model versions
@task(container_image=image, cache=True, cache_version="1.0")
def download_model(
    model: str,
) -> Annotated[FlyteDirectory, MyOllamaModel(model=Inputs.model)]:
    """
    Downloads a model using Ollama. This publishes the model as an artifact.

    :param model: Model to download
    :return: Directory containing the model
    """

    subprocess.Popen(["ollama", "serve"])
    flytekit.current_context().logging.info("Waiting for Ollama service to be ready")
    wait_for_ollama()

    flytekit.current_context().logging.info("Pulling model")
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)

    flytekit.current_context().logging.info("Stopping Ollama service")
    subprocess.run(["ollama", "stop"])

    flytekit.current_context().logging.info("Exporting model")
    print("Check models")
    model_dir = Path("/home/flytekit/.ollama/models")
    for path in model_dir.glob("*/**"):
        print(path)
    return model_dir


@workflow
def download_multiple_models(models: typing.List[str] = ["llama3.1"]) -> typing.List[FlyteDirectory]:
    """
    Downloads multiple models using Ollama. This publishes each model as a separate artifact.
    """
    return flytekit.map_task(download_model)(model=models)
