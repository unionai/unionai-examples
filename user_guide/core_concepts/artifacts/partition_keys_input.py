from datetime import datetime

import pandas as pd
from flytekit import ImageSpec, task, workflow
from flytekit.core.artifact import Artifact, Inputs, Granularity
from typing_extensions import Annotated

pandas_image = ImageSpec(
    packages=["pandas==2.2.2"]
)

BasicArtifact = Artifact(
    name="my_basic_artifact",
    time_partitioned=True,
    time_partition_granularity=Granularity.HOUR,
    partition_keys=["key1"]
)


@task(container_image=pandas_image)
def t1(
    key1: str, dt: datetime
) -> Annotated[pd.DataFrame, BasicArtifact(key1=Inputs.key1)]:
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    return BasicArtifact.create_from(
        df,
        time_partition=dt,
        key1=key1
    )


@workflow
def wf(dt: datetime = datetime.fromtimestamp(1887639468), val: str = "hello"):
    t1(key1=val, dt=dt)
