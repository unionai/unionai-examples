from typing import Any, List
from flytekit.configuration import ImageConfig
from unionai.remote import UnionRemote
from flytekit.tools.translator import Options
from workflow.foo import foo_wf


def run_workflow():
    options = Options()
    ur = UnionRemote()
    ur.register_script(
        entity=foo_wf,
        source_path="./workflow",
        options=options,
        version="v2",
        copy_all=True
    )
#    ur.execute(
#        entity=hello_wf,
#        image_config=ImageConfig.auto_default_image(),
#        options=options,
#        version="v2",
#        wait=False,
#        inputs=None,
#        execution_name="id1",
#    )


if __name__ == "__main__":
    run_workflow()
