from flytekit import kwtypes
from flytekit.types.file import FlyteFile
from flytekitplugins.awssagemaker_inference import (
    create_sagemaker_deployment,
    triton_image_uri,
)

NEW_DEPLOYMENT_NAME = "{inputs.deployment_name}-{idempotence_token}"
EXISTING_DEPLOYMENT_NAME = "{inputs.deployment_name}-{inputs.idempotence_token}"

sd_deployment = create_sagemaker_deployment(
    name="stable-diffusion",
    model_input_types=kwtypes(
        deployment_name=str, model_path=FlyteFile, execution_role_arn=str
    ),
    model_config={
        "ModelName": NEW_DEPLOYMENT_NAME,
        "PrimaryContainer": {
            "Image": "{images.sd_deployment_image}",
            "ModelDataUrl": "{inputs.model_path}",
            "Environment": {
                "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "pipeline",
                "SAGEMAKER_TRITON_LOG_INFO": "false --load-model=text_encoder --load-model=vae",
            },
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=kwtypes(
        deployment_name=str,
        initial_instance_count=int,
        instance_type=str,
    ),
    endpoint_config_config={
        "EndpointConfigName": NEW_DEPLOYMENT_NAME,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": EXISTING_DEPLOYMENT_NAME,
                "InitialInstanceCount": "{inputs.initial_instance_count}",
                "InstanceType": "{inputs.instance_type}",
            },
        ],
    },
    endpoint_input_types=kwtypes(deployment_name=str),
    endpoint_config={
        "EndpointName": NEW_DEPLOYMENT_NAME,
        "EndpointConfigName": EXISTING_DEPLOYMENT_NAME,
    },
    images={"sd_deployment_image": triton_image_uri(version="23.12")},
    region_at_runtime=True,
)
