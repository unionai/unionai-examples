import logging
import os
import random
import string
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, List
from flytekit.configuration import ImageConfig, PlatformConfig
from union.remote import UnionRemote
from flytekit.tools.translator import Options


def load_function(python_file: Path, function: str) -> Any:
    module_path = python_file.parent.resolve()
    module = os.path.basename(python_file).replace(".py", "")
    sys.path.append(str(module_path))
    module = import_module(module)
    return getattr(module, function)


def run_workflow(
    workflow_file: str,
    workflow: str,
):
    try:
        options = Options()

        workflow_file_path = Path(workflow_file)

        ur = UnionRemote()
        entity_wf = load_function(workflow_file_path, workflow)
        random_suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))  # noqa: S311
        execution_id = f"{workflow.replace('_', '-')}-{random_suffix}"

        ur.register_script(
            entity=entity_wf,
            source_path=workflow_file_path.parent.resolve(),
            options=options,
            version=random_suffix,
            copy_all=True,
        )

        ur.execute(
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
        workflow_file="workflow/hello_world.py",
        workflow="hello_world_wf",
    )
