import os

from flytekit import kwtypes, workflow
from flytekitplugins.awssagemaker_inference import (
    SageMakerInvokeEndpointTask,
    create_sagemaker_deployment,
    delete_sagemaker_deployment,
    triton_image_uri,
)

MODEL_NAME = "triton-resnet-trt"
ENDPOINT_NAME = "triton-resnet-trt-endpoint"
ENDPOINT_CONFIG_NAME = "triton-resnet-trt-endpoint-config"
REGION = "us-east-2"
INSTANCE = "ml.g4dn.4xlarge"


sagemaker_deployment_wf = create_sagemaker_deployment(
    name="triton-tensorrt",
    model_input_types=kwtypes(model_path=str, execution_role_arn=str),
    model_config={
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "{images.triton_image}",
            "ModelDataUrl": "{inputs.model_path}",
            "Environment": {"SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "resnet"},
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=kwtypes(instance_type=str),
    endpoint_config_config={
        "EndpointConfigName": ENDPOINT_CONFIG_NAME,
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": MODEL_NAME,
                "InitialInstanceCount": 1,
                "InstanceType": "{inputs.instance_type}",
            },
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {"S3OutputPath": os.getenv("TRITON_S3_OUTPUT_PATH")}
        },
    },
    endpoint_config={
        "EndpointName": ENDPOINT_NAME,
        "EndpointConfigName": ENDPOINT_CONFIG_NAME,
    },
    images={"triton_image": triton_image_uri},
    region=REGION,
)


@workflow
def model_deployment_workflow(
    model_path: str = os.getenv("TRITON_TENSORRT_MODEL_DATA_URL"),
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
) -> str:
    return sagemaker_deployment_wf(
        model_path=model_path,
        execution_role_arn=execution_role_arn,
        instance_type=INSTANCE,
    )


########################
# INVOKE ENDPOINT TASK #
########################
invoke_endpoint = SageMakerInvokeEndpointTask(
    name="sagemaker-triton-tensorrt-invoke-endpoint",
    config={
        "EndpointName": ENDPOINT_NAME,
        "InputLocation": os.getenv("TRITON_TENSORRT_INPUT_LOCATION"),
        "ContentType": "application/octet-stream",
    },
    region=REGION,
)


################################
# DEPLOYMENT DELETION WORKFLOW #
################################
sagemaker_deployment_deletion_wf = delete_sagemaker_deployment(
    name="sagemaker-triton-deployment-deletion", region=REGION
)


@workflow
def deployment_deletion_workflow():
    sagemaker_deployment_deletion_wf(
        endpoint_name=ENDPOINT_NAME,
        endpoint_config_name=ENDPOINT_CONFIG_NAME,
        model_name=MODEL_NAME,
    )
