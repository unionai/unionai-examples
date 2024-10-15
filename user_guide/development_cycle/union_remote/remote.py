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
        copy_all=True
    )
    ur.execute(
        entity=foo_wf,
        inputs={}
    )


if __name__ == "__main__":
    run_workflow()
