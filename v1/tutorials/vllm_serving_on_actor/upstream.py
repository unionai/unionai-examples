import random
from typing import Annotated
from flytekit import task, workflow
from utils import TextSample, TextSampleArtifact


@task
def get_text() -> TextSample:
    text_samples = {
        "1": "Elon Musk, the CEO of Tesla, announced a partnership with SpaceX to launch satellites from Cape Canaveral in 2024.",
        "2": "On September 15th, 2023, Serena Williams won the U.S. Open at Arthur Ashe Stadium in New York City.",
        "3": "President Joe Biden met with leaders from NATO in Brussels to discuss the conflict in Ukraine on July 10th, 2022.",
        "4": "Sundar Pichai, the CEO of Google, gave the keynote speech at the Google I/O conference held in Mountain View on May 11th, 2023.",
        "5": "J.K. Rowling, author of the Harry Potter series, gave a talk at Oxford University in December 2019.",
    }
    id = random.choice(list(text_samples.keys()))
    return TextSample(id=id, body=text_samples[id])


@workflow
def upstream_wf() -> Annotated[TextSample, TextSampleArtifact]:
    return get_text()
