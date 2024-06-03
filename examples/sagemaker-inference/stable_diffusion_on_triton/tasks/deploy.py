from flytekit import kwtypes
from flytekit.types.file import FlyteFile
from flytekitplugins.awssagemaker_inference import (
    create_sagemaker_deployment,
    triton_image_uri,
)

sd_deployment = create_sagemaker_deployment(
    name="stable-diffusion",
    model_input_types=kwtypes(
        model_name=str, model_path=FlyteFile, execution_role_arn=str
    ),
    model_config={
        "ModelName": "{inputs.model_name}",
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
        model_name=str,
        initial_instance_count=int,
        instance_type=str,
        endpoint_config_name=str,
    ),
    endpoint_config_config={
        "EndpointConfigName": "{inputs.endpoint_config_name}",
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": "{inputs.model_name}",
                "InitialInstanceCount": "{inputs.initial_instance_count}",
                "InstanceType": "{inputs.instance_type}",
            },
        ],
    },
    endpoint_input_types=kwtypes(endpoint_name=str, endpoint_config_name=str),
    endpoint_config={
        "EndpointName": "{inputs.endpoint_name}",
        "EndpointConfigName": "{inputs.endpoint_config_name}",
    },
    images={"sd_deployment_image": triton_image_uri(version="23.12")},
    region_at_runtime=True,
)
