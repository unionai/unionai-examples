import os

from flytekit import workflow

from stable_diffusion_on_triton.tasks.deploy import sd_deployment
from stable_diffusion_on_triton.tasks.optimize import compress_model, optimize_model


@workflow
def stable_diffusion_on_triton_wf(
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    repo_id: str = "CompVis/stable-diffusion-v1-4",
    model_name: str = "stable-diffusion-model-non-finetuned",
    endpoint_config_name: str = "stable-diffusion-endpoint-config-non-finetuned",
    endpoint_name: str = "stable-diffusion-endpoint-non-finetuned",
    instance_type: str = "ml.g5.2xlarge",  # A10G used for model compilation
    initial_instance_count: int = 1,
    region: str = "us-east-2",
) -> str:
    model_repo = optimize_model(fused_model_name=repo_id)
    compressed_model = compress_model(model_repo=model_repo)
    deployment = sd_deployment(
        model_name=model_name,
        endpoint_config_name=endpoint_config_name,
        endpoint_name=endpoint_name,
        model_path=compressed_model,
        execution_role_arn=execution_role_arn,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        region=region,
    )
    return deployment
