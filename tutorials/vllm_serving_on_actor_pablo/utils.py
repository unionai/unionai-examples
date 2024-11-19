from dataclasses import dataclass
from flytekit import Artifact


@dataclass
class TextSample:
    id: str
    body: str


TextSampleArtifact = Artifact(name="text_sample")
