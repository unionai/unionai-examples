from datetime import timedelta
from flytekit import LaunchPlan, FixedRate
from union.artifacts import OnArtifact

from ner import TextSample, ner_wf, upstream_wf

lp = LaunchPlan.get_or_create(
    workflow=ner_wf, name="ner_lp_v4", trigger=OnArtifact(trigger_on=TextSample)
)

upstream_lp = LaunchPlan.get_or_create(
    workflow=upstream_wf, name="upstream_lp", schedule=FixedRate(duration=timedelta(minutes=1))
)
