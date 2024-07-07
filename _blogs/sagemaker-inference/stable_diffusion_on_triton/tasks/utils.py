from flytekit.core.artifact import Artifact

ModelArtifact = Artifact(
    name="stable-diffusion-fine-tuned", partition_keys=["dataset", "type"]
)
# ModelArtifact = Artifact(name="stable-diffusion-fine-tuned")
