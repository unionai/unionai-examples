import os

from flytekit import kwtypes, workflow
from flytekitplugins.awssagemaker_inference import (
    SagemakerInvokeEndpointTask,
    create_sagemaker_deployment,
    delete_sagemaker_deployment,
)

MODEL_NAME = "triton-resnet-pt"
ENDPOINT_NAME = "triton-resnet-pt-endpoint"
ENDPOINT_CONFIG_NAME = "triton-resnet-pt-endpoint-config"
REGION = "us-east-2"
INSTANCE = "ml.g4dn.4xlarge"

account_id_map = {
    "us-east-1": "785573368785",
    "us-east-2": "007439368137",
    "us-west-1": "710691900526",
    "us-west-2": "301217895009",
    "eu-west-1": "802834080501",
    "eu-west-2": "205493899709",
    "eu-west-3": "254080097072",
    "eu-north-1": "601324751636",
    "eu-south-1": "966458181534",
    "eu-central-1": "746233611703",
    "ap-east-1": "110948597952",
    "ap-south-1": "763008648453",
    "ap-northeast-1": "941853720454",
    "ap-northeast-2": "151534178276",
    "ap-southeast-1": "324986816169",
    "ap-southeast-2": "355873309152",
    "cn-northwest-1": "474822919863",
    "cn-north-1": "472730292857",
    "sa-east-1": "756306329178",
    "ca-central-1": "464438896020",
    "me-south-1": "836785723513",
    "af-south-1": "774647643957",
}


base = "amazonaws.com.cn" if REGION.startswith("cn-") else "amazonaws.com"
triton_image_uri = (
    "{account_id}.dkr.ecr.{region}.{base}/sagemaker-tritonserver:21.08-py3".format(
        account_id=account_id_map[REGION], region=REGION, base=base
    )
)


sagemaker_deployment_wf = create_sagemaker_deployment(
    name="sagemaker-deployment-triton",
    model_input_types=kwtypes(model_path=str, execution_role_arn=str),
    model_config={
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "{container.image}",
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
    container_image=triton_image_uri,
    region=REGION,
)


@workflow
def model_deployment_workflow(
    model_path: str = os.getenv("TRITON_MODEL_DATA_URL"),
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
invoke_endpoint = SagemakerInvokeEndpointTask(
    name="sagemaker-triton-invoke-endpoint",
    config={
        "EndpointName": ENDPOINT_NAME,
        "InputLocation": os.getenv("TRITON_INPUT_LOCATION"),
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
