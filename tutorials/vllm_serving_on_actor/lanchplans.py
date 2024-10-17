from datetime import timedelta
from flytekit import LaunchPlan, FixedRate
from union.artifacts import OnArtifact

from ner import ner_wf
from utils import TextSampleArtifact
from upstream import upstream_wf

ner_lp = LaunchPlan.get_or_create(
    workflow=ner_wf, name="ner_lp", trigger=OnArtifact(trigger_on=TextSampleArtifact)
)

upstream_lp = LaunchPlan.get_or_create(
    workflow=upstream_wf, name="upstream_lp", schedule=FixedRate(duration=timedelta(minutes=1))
)
