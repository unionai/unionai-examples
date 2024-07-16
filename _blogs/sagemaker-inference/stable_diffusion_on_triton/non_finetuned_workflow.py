import os

from flytekit import workflow

from stable_diffusion_on_triton.tasks.deploy import sd_deployment
from stable_diffusion_on_triton.tasks.non_finetuned_optimize import (
    compress_model_non_finetuned,
    optimize_model_non_finetuned,
)


@workflow
def stable_diffusion_on_triton_wf(
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    repo_id: str = "CompVis/stable-diffusion-v1-4",
    deployment_name: str = "stable-diffusion-pokemon",
    instance_type: str = "ml.g5.2xlarge",  # using A10G for model compilation
    initial_instance_count: int = 1,
    region: str = "us-east-2",
) -> list[dict]:
    model_repo = optimize_model_non_finetuned(model=repo_id)
    compressed_model = compress_model_non_finetuned(model_repo=model_repo)
    deployment = sd_deployment(
        deployment_name=deployment_name,
        model_path=compressed_model,
        execution_role_arn=execution_role_arn,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        region=region,
    )
    return deployment
