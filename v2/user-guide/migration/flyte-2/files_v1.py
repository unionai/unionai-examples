import os

from flytekit import task, workflow, current_context
from flytekit.types.file import FlyteFile


@task
def write_file(content: str) -> FlyteFile:
    path = os.path.join(current_context().working_directory, "out.txt")
    with open(path, "w") as f:
        f.write(content)
    return FlyteFile(path=path)


@task
def read_file(f: FlyteFile) -> str:
    with open(f.download()) as fh:
        return fh.read()


@workflow
def main(content: str) -> str:
    f = write_file(content=content)
    return read_file(f=f)
