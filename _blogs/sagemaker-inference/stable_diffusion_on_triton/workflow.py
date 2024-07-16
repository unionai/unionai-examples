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
    deployment_name: str = "stable-diffusion-lora-pokemon",
    instance_type: str = "ml.g5.2xlarge",  # using A10G for model compilation
    initial_instance_count: int = 1,
    region: str = "us-east-2",
) -> list[dict]:
    lora = stable_diffusion_finetuning(args=finetuning_args)
    fused_lora = fuse_lora(
        repo_id=finetuning_args.pretrained_model_name_or_path,
        lora=lora,
        dataset=finetuning_args.dataset_name,
    )
    model_repo = optimize_model(
        fused_lora=fused_lora, dataset=finetuning_args.dataset_name
    )
    compressed_model = compress_model(
        model_repo=model_repo, dataset=finetuning_args.dataset_name
    )
    deployment = sd_deployment(
        deployment_name=deployment_name,
        model_path=compressed_model,  # model_path = ModelArtifact.query(dataset=Inputs.dataset, type="sagemaker-compressed-model")
        execution_role_arn=execution_role_arn,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        region=region,
    )
    return deployment
