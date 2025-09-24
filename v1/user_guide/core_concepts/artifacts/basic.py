import pandas as pd
from flytekit import ImageSpec, task, workflow
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated

pandas_image = ImageSpec(
    packages=["pandas==2.2.2"]
)

BasicTaskData = Artifact(
    name="my_basic_artifact"
)


@task(container_image=pandas_image)
def t1() -> Annotated[pd.DataFrame, BasicTaskData]:
    my_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    return BasicTaskData.create_from(my_df)


@workflow
def wf() -> pd.DataFrame:
    return t1()
