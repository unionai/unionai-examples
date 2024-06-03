import os

from flytekit import ImageSpec, kwtypes, workflow
from flytekitplugins.awssagemaker_inference import (
    SageMakerDeleteEndpointConfigTask,
    SageMakerDeleteEndpointTask,
    SageMakerDeleteModelTask,
    SageMakerEndpointConfigTask,
    SageMakerEndpointTask,
    SageMakerInvokeEndpointTask,
    SageMakerModelTask,
    create_sagemaker_deployment,
    delete_sagemaker_deployment,
)

REGION = "us-east-2"
MODEL_NAME = "xgboost"
ENDPOINT_CONFIG_NAME = "xgboost-endpoint-config"
ENDPOINT_NAME = "xgboost-endpoint"


custom_image = ImageSpec(
    name="sagemaker-xgboost",
    registry=os.getenv("REGISTRY"),
    requirements="requirements.txt",
    apt_packages=["git"],
    source_root="fastapi",
).with_commands(["chmod +x /root/serve"])


####################
# DEPLOYMENT TASKS #
####################
create_sagemaker_model = SageMakerModelTask(
    name="sagemaker_model",
    config={
        "ModelName": "{inputs.model_name}",
        "PrimaryContainer": {
            "Image": "{images.primary_container_image}",
            "ModelDataUrl": "{inputs.model_data_url}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    images={"primary_container_image": custom_image},
    inputs=kwtypes(model_name=str, model_data_url=str, execution_role_arn=str),
    region=REGION,
)

create_endpoint_config = SageMakerEndpointConfigTask(
    name="sagemaker_endpoint_config",
    config={
        "EndpointConfigName": "{inputs.endpoint_config_name}",
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": "{inputs.model_name}",
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m4.xlarge",
            },
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {"S3OutputPath": "{inputs.s3_output_path}"}
        },
    },
    region=REGION,
    inputs=kwtypes(endpoint_config_name=str, model_name=str, s3_output_path=str),
)

create_endpoint = SageMakerEndpointTask(
    name="sagemaker_endpoint",
    config={
        "EndpointName": "{inputs.endpoint_name}",
        "EndpointConfigName": "{inputs.endpoint_config_name}",
    },
    region=REGION,
    inputs=kwtypes(endpoint_name=str, endpoint_config_name=str),
)


delete_endpoint = SageMakerDeleteEndpointTask(
    name="sagemaker_delete_endpoint",
    config={"EndpointName": "{inputs.endpoint_name}"},
    region=REGION,
    inputs=kwtypes(endpoint_name=str),
)

delete_endpoint_config = SageMakerDeleteEndpointConfigTask(
    name="sagemaker_delete_endpoint_config",
    config={"EndpointConfigName": "{inputs.endpoint_config_name}"},
    region=REGION,
    inputs=kwtypes(endpoint_config_name=str),
)

delete_model = SageMakerDeleteModelTask(
    name="sagemaker_delete_model",
    config={"ModelName": "{inputs.model_name}"},
    region=REGION,
    inputs=kwtypes(model_name=str),
)


@workflow
def example_workflow(
    model_name: str = MODEL_NAME,
    model_data_url: str = os.getenv("MODEL_DATA_URL"),
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    s3_output_path: str = os.getenv("S3_OUTPUT_PATH"),
    endpoint_config_name: str = ENDPOINT_CONFIG_NAME,
    endpoint_name: str = ENDPOINT_NAME,
):
    create_sagemaker_model(
        model_name=model_name,
        model_data_url=model_data_url,
        execution_role_arn=execution_role_arn,
    )
    create_endpoint_config(
        endpoint_config_name=endpoint_config_name,
        model_name=model_name,
        s3_output_path=s3_output_path,
    )
    create_endpoint(
        endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name
    )
    delete_endpoint(endpoint_name=endpoint_name)
    delete_endpoint_config(endpoint_config_name=endpoint_config_name)
    delete_model(model_name=model_name)


#######################
# DEPLOYMENT WORKFLOW #
#######################
sagemaker_deployment_wf = create_sagemaker_deployment(
    name="xgboost",
    model_input_types=kwtypes(model_path=str, execution_role_arn=str),
    model_config={
        "ModelName": MODEL_NAME,
        "PrimaryContainer": {
            "Image": "{images.primary_container_image}",
            "ModelDataUrl": "{inputs.model_path}",
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
            "OutputConfig": {
                "S3OutputPath": "s3://sagemaker-agent-xgboost/inference-output/output"
            }
        },
    },
    endpoint_config={
        "EndpointName": ENDPOINT_NAME,
        "EndpointConfigName": ENDPOINT_CONFIG_NAME,
    },
    images={"primary_container_image": custom_image},
    region=REGION,
)


@workflow
def model_deployment_workflow(
    execution_role_arn: str,
    model_path: str = "s3://sagemaker-agent-xgboost/model.tar.gz",
) -> str:
    return sagemaker_deployment_wf(
        model_path=model_path,
        execution_role_arn=execution_role_arn,
        instance_type="ml.m4.xlarge",
    )


########################
# INVOKE ENDPOINT TASK #
########################
invoke_endpoint = SageMakerInvokeEndpointTask(
    name="sagemaker_invoke_endpoint",
    config={
        "EndpointName": ENDPOINT_NAME,
        "InputLocation": os.getenv("INPUT_LOCATION"),
    },
    region=REGION,
)


################################
# DEPLOYMENT DELETION WORKFLOW #
################################
sagemaker_deployment_deletion_wf = delete_sagemaker_deployment(
    name="xgboost", region=REGION
)


@workflow
def deployment_deletion_workflow():
    sagemaker_deployment_deletion_wf(
        endpoint_name=ENDPOINT_NAME,
        endpoint_config_name=ENDPOINT_CONFIG_NAME,
        model_name=MODEL_NAME,
    )
