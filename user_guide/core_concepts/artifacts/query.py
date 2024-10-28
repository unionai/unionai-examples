from datetime import datetime

import pandas as pd
from flytekit import ImageSpec, task, workflow
from flytekit.core.artifact import Artifact, Inputs

pandas_image = ImageSpec(
    packages=["pandas==2.2.2"]
)

BasicArtifact = Artifact(
    name="my_basic_artifact"
)


@task(container_image=pandas_image)
def t1(key1: str, dt: datetime, data: pd.DataFrame):
    print(f"key1: {key1}")
    print(f"Date: {dt}")
    print(f"Data retrieved from query: {data}")


data_query = BasicArtifact.query(
    time_partition=Inputs.dt,
    key1=Inputs.key1,
)


@workflow
def query_wf(
    key1: str,
    dt: datetime,
    data: pd.DataFrame = data_query
):
    t1(key1=key1, dt=dt, data=data)
