from datetime import timedelta

from flytekit import approve, workflow

from .tasks.batch_predict import batch_predict
from .tasks.compress_model import compress_model
from .tasks.deploy import sam_deployment
from .tasks.fine_tune import fine_tune_sam


@workflow
def sam_sagemaker_deployment(
    execution_role_arn: str,
    dataset_name: str = "nielsr/breast-cancer",
    model_name: str = "sam-model",
    endpoint_config_name: str = "sam-endpoint-config",
    endpoint_name: str = "sam-endpoint",
    instance_type: str = "ml.m4.xlarge",
    initial_instance_count: int = 1,
    region: str = "us-east-2",
    output_path: str = "s3://sagemaker-sam/inference-output/output",
) -> str:
    model = fine_tune_sam(dataset_name=dataset_name)
    predictions = batch_predict(model=model)

    approve_filter = approve(
        predictions, "batch_predictions_approval", timeout=timedelta(hours=2)
    )

    compressed_model = compress_model(model=model)
    approve_filter >> compressed_model

    deployment = sam_deployment(
        model_name=model_name,
        endpoint_config_name=endpoint_config_name,
        endpoint_name=endpoint_name,
        model_path=compressed_model,
        execution_role_arn=execution_role_arn,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        region=region,
        output_path=output_path,
    )

    return deployment
