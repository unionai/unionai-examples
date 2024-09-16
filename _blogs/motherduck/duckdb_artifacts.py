from flytekit import Artifact
from union.artifacts import OnArtifact

RecentEcommerceData = Artifact(
    name="recent_ecommerce_data"
)

on_recent_ecommerce_data = OnArtifact(
    trigger_on=RecentEcommerceData,
)