from datetime import datetime

import pandas as pd
from flytekit import ImageSpec, task, workflow
from flytekit.core.artifact import Artifact, Granularity
from typing_extensions import Annotated

pandas_image = ImageSpec(
    packages=["pandas==2.2.2", "pyarrow"]
)

BasicArtifact = Artifact(
    name="my_basic_artifact",
    time_partitioned=True,
    time_partition_granularity=Granularity.HOUR
)


@task(container_image=pandas_image)
def t1(date: datetime)\
     -> Annotated[pd.DataFrame, BasicArtifact]:
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    return BasicArtifact.create_from(df, time_partition=date)


@workflow
def wf(run_date: datetime = datetime.fromtimestamp(1887639468)):
    return t1(date=run_date)
