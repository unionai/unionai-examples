from flytekit.core.artifact import Artifact

ModelArtifact = Artifact(
    name="fine-tuned-stable-diffusion", partition_keys=["dataset", "type"]
)
