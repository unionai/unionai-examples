import os

from flytekit import workflow

from stable_diffusion_on_triton.tasks.deploy import sd_deployment
from stable_diffusion_on_triton.tasks.fine_tune import (
    FineTuningArgs,
    stable_diffusion_finetuning,
)
from stable_diffusion_on_triton.tasks.fuse_lora import fuse_lora
from stable_diffusion_on_triton.tasks.optimize import compress_model, optimize_model


@workflow
def stable_diffusion_on_triton_wf(
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    finetuning_args: FineTuningArgs = FineTuningArgs(),
    fused_model_name: str = "Samhita/fused-stable-diffusion-lora",
    model_name: str = "stable-diffusion-model",
    endpoint_config_name: str = "stable-diffusion-endpoint-config",
    endpoint_name: str = "stable-diffusion-endpoint",
    instance_type: str = "ml.g5.2xlarge",  # A10G used for model compilation
    initial_instance_count: int = 1,
    region: str = "us-east-2",
) -> str:
    repo_id = stable_diffusion_finetuning(args=finetuning_args)
    fuse_lora_output = fuse_lora(
        model_name=finetuning_args.pretrained_model_name_or_path,
        repo_id=repo_id,
        fused_model_name=fused_model_name,
    )
    model_repo = optimize_model(fused_model_name=fuse_lora_output)
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
