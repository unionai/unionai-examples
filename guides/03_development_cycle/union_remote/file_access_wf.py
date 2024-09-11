from flytekit import task, workflow
from flytekit.types.file import FlyteFile


@task
def test_file_read(n: FlyteFile) -> str:
    file = open(n, "r")
    content = file.read()
    print(content)
    file.close()
    return content


@workflow
def wf(n: FlyteFile="test.txt")->str:
    return test_file_read(n=n)