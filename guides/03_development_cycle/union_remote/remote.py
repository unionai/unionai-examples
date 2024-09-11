import json
import logging
import os
import random
import string
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, List

import keyring
from flytekit import WorkflowExecutionPhase
from flytekit.configuration import Config as FlyteConfig
from flytekit.configuration import ImageConfig, PlatformConfig
from flytekit.core.notification import Email
from flytekit.models.security import Identity, SecurityContext
from flytekit.remote.remote import FlyteRemote
from flytekit.tools.translator import Options

def _create_flyte_remote(project: str, domain: str, url: str) -> FlyteRemote:
    """Creates a flyte config file."""
    return FlyteRemote(
        config=FlyteConfig(platform=PlatformConfig(endpoint=url, insecure=True)),
        default_domain=domain,
        default_project=project,
    )

def load_function(python_file: Path, function: str) -> Any:
    """Loads a function by name."""
    module_path = python_file.parent.resolve()
    module = os.path.basename(python_file).replace(".py", "")

    sys.path.append(str(module_path))
    module = import_module(module)

    return getattr(module, function)


def run_workflow(
    project: str,
    domain: str,
    url: str,
    workflow_file: str,
    workflow: str,
):
    """Runs a Flyte workflow."""
    try:
        options = Options()

        workflow_file_path = Path(workflow_file)

        flyte_remote = _create_flyte_remote(project=project, domain=domain, url=url)
        entity_wf = load_function(workflow_file_path, workflow)
        random_suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))  # noqa: S311
        execution_id = f"{workflow.replace('_', '-')}-{random_suffix}"

        flyte_remote.register_script(
            entity=entity_wf,
            source_path=workflow_file_path.parent.resolve(),
            options=options,
            version=random_suffix,
            copy_all=True,
        )

        flyte_remote.execute(
            entity=entity_wf,
            image_config=ImageConfig.auto_default_image(),
            options=options,
            version=random_suffix,
            wait=False,
            inputs=None,
            execution_name=execution_id,
        )

        logging.info(f"Flyte execution ID: {execution_id}")
    except Exception as e:
        logging.error(f"Error in Flyte Runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_workflow(
        project="flytesnacks",
        domain="development",
        url="",
        workflow_file="hello_world.py",
        workflow="hello_world_wf",
    )