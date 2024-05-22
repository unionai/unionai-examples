import pandas as pd
from flytekit import task, workflow
from flytekit.core.artifact import Artifact
from unionai.artifacts import ModelCard
from typing_extensions import Annotated

BasicArtifact = Artifact(name="my_basic_artifact")


def generate_md_contents(df: pd.DataFrame) -> str:
    contents = "# Dataset Card\n" "\n" "## Tabular Data\n"
    contents = contents + df.to_markdown()
    return contents


@task
def t1() -> Annotated[pd.DataFrame, BasicArtifact]:
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    return BasicArtifact.create_from(
        df,
        ModelCard(generate_md_contents(df))
    )


@workflow
def wf():
    t1()
