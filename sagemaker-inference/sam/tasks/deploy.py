import os

from flytekit import ImageSpec, kwtypes
from flytekit.types.file import FlyteFile
from flytekitplugins.awssagemaker_inference import (
    SageMakerInvokeEndpointTask,
    create_sagemaker_deployment,
)

sam_deployment_image = ImageSpec(
    name="sam-deployment",
    registry=os.getenv("REGISTRY"),
    packages=[
        "transformers==4.38.2",
        "torch==2.2.1",
        "monai==1.3.0",
        "matplotlib==3.8.3",
        "fastapi==0.110.0",
        "uvicorn==0.29.0",
    ],
    source_root="sam/tasks/fastapi",
).with_commands(["chmod +x /root/serve"])


sam_deployment = create_sagemaker_deployment(
    name="sam",
    model_input_types=kwtypes(
        model_name=str, model_path=FlyteFile, execution_role_arn=str
    ),
    model_config={
        "ModelName": "{inputs.model_name}",
        "PrimaryContainer": {
            "Image": "{images.sam_deployment_image}",
            "ModelDataUrl": "{inputs.model_path}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=kwtypes(
        model_name=str,
        initial_instance_count=int,
        instance_type=str,
        endpoint_config_name=str,
        output_path=str,
    ),
    endpoint_config_config={
        "EndpointConfigName": "{inputs.endpoint_config_name}",
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": "{inputs.model_name}",
                "InitialInstanceCount": "{inputs.initial_instance_count}",
                "InstanceType": "{inputs.instance_type}",
            },
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {"S3OutputPath": "{inputs.output_path}"}
        },
    },
    endpoint_input_types=kwtypes(endpoint_name=str, endpoint_config_name=str),
    endpoint_config={
        "EndpointName": "{inputs.endpoint_name}",
        "EndpointConfigName": "{inputs.endpoint_config_name}",
    },
    images={"sam_deployment_image": sam_deployment_image},
    region_at_runtime=True,
)


invoke_endpoint = SageMakerInvokeEndpointTask(
    name="sam-invoke-endpoint",
    config={
        "EndpointName": "{inputs.endpoint_name}",
        "InputLocation": "s3://sagemaker-sam/inference_input",
        "ContentType": "application/octet-stream",
    },
    inputs=kwtypes(endpoint_name=str, region=str),
)
