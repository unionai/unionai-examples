from flytekit import task, ImageSpec
from flytekit.core.artifact import Artifact
from typing_extensions import Annotated
from union.artifacts import ModelCard
from tasks.query_calls import CallData
from typing import List
import os
import json
import pandas as pd


generate_corpus_img = ImageSpec(
    packages=[
        "flytekit==1.13.0",
        "union==0.1.48",
        "pandas==2.2.2",
        "tabulate==0.9.0"
    ],
    registry=os.getenv("DOCKER_REGISTRY")
)


CallDataCorpus = Artifact(name="call_data_corpus")


@task(
    container_image=generate_corpus_img,
)
def generate_corpus(prev_data: List[CallData], new_data: List[CallData]) -> Annotated[list[CallData], CallDataCorpus]:
    df_dict = {
        "date": [],
        "title": [],
        "url": []
    }
    all_calls = prev_data + new_data
    for call_data in all_calls:
        call_data.call_metadata.download()
        with open(call_data.call_metadata.path, 'r') as file:
            data = json.load(file)

        meta_data = data.get("metaData", {})
        df_dict["date"].append(meta_data.get("scheduled", ""))
        df_dict["title"].append(meta_data.get("title", "").replace("|", ""))
        df_dict["url"].append(meta_data.get("url", ""))

    card_df = pd.DataFrame(df_dict)

    card_df['date'] = pd.to_datetime(card_df['date'], format='ISO8601')
    card_df = card_df.sort_values(by='date', ascending=False)
    card_df = card_df.reset_index(drop=True)

    return CallDataCorpus.create_from(
        prev_data,
        ModelCard(card_df.to_markdown())
    )
