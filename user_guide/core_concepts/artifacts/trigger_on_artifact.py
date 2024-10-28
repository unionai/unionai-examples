from datetime import datetime

import pandas as pd
from flytekit import ImageSpec, task, workflow, LaunchPlan
from flytekit.core.artifact import Artifact, Inputs
from typing_extensions import Annotated
from union.artifacts import OnArtifact

pandas_image = ImageSpec(
    packages=["pandas==2.2.2"]
)

UpstreamArtifact = Artifact(
    name="my_upstream_artifact",
    time_partitioned=True,
    partition_keys=["key1"],
)


@task(container_image=pandas_image)
def upstream_t1(key1: str) -> Annotated[pd.DataFrame,
                                        UpstreamArtifact(key1=Inputs.key1)]:
    dt = datetime.now()
    my_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    return UpstreamArtifact.create_from(my_df, key1=key1,
                                        time_partition=dt)


@workflow
def upstream_wf() -> pd.DataFrame:
    return upstream_t1(key1="value1")


on_upstream_artifact = OnArtifact(
    trigger_on=UpstreamArtifact,
)


@task
def downstream_t1():
    print("Downstream task triggered")


@workflow
def downstream_wf():
    downstream_t1()


downstream_triggered = LaunchPlan.create(
    "downstream_with_trigger_lp",
    downstream_wf,
    trigger=on_upstream_artifact
)
