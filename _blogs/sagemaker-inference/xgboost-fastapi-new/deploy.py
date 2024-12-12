import os
import tarfile

import flytekit as fl
from flytekit.types.file import FlyteFile
from flytekitplugins.awssagemaker_inference import create_sagemaker_deployment
from numpy import loadtxt

DEPLOYMENT_NAME = "{inputs.deployment_name}"


image = fl.ImageSpec(
    name="sagemaker-xgboost",
    registry=os.getenv("REGISTRY", "ghcr.io/unionai-oss"),
    packages=["xgboost", "scikit-learn", "flytekitplugins-awssagemaker"],
    source_root="fastapi",
).with_commands(["chmod +x /root/serve"])


deployment_image = fl.ImageSpec(
    name="sagemaker-xgboost",
    registry=os.getenv("AWS_REGISTRY"),
    packages=["xgboost", "fastapi", "uvicorn", "scikit-learn"],
    source_root="fastapi",
    env={"PATH": "/root:$PATH"},
).with_commands(["chmod +x /root/serve"])


@fl.task(cache=True, cache_version="0.1", container_image=image)
def train_model(dataset: FlyteFile) -> FlyteFile:
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    dataset = loadtxt(dataset.download(), delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=7)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    serialized_model = os.path.join(
        fl.current_context().working_directory, "xgboost_model.json"
    )
    booster = model.get_booster()
    booster.save_model(serialized_model)

    return FlyteFile(path=serialized_model)


@fl.task(cache=True, cache_version="0.1", container_image=image)
def convert_to_tar(model: FlyteFile) -> FlyteFile:
    tf = tarfile.open("model.tar.gz", "w:gz")
    tf.add(model.download(), arcname="xgboost_model")
    tf.close()

    return FlyteFile("model.tar.gz")


sagemaker_deployment_wf = create_sagemaker_deployment(
    name="xgboost",
    model_input_types=fl.kwtypes(deployment_name=str, model_path=FlyteFile, execution_role_arn=str),
    model_config={
        "ModelName": DEPLOYMENT_NAME,
        "PrimaryContainer": {
            "Image": "{images.primary_container_image}",
            "ModelDataUrl": "{inputs.model_path}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=fl.kwtypes(deployment_name=str, initial_instance_count=int, instance_type=str),
    endpoint_config_config={
        "EndpointConfigName": DEPLOYMENT_NAME,
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": DEPLOYMENT_NAME,
                "InitialInstanceCount": "{inputs.initial_instance_count}",
                "InstanceType": "{inputs.instance_type}",
            },
        ],
    },
    endpoint_input_types=fl.kwtypes(deployment_name=str),
    endpoint_config={
        "EndpointName": DEPLOYMENT_NAME,
        "EndpointConfigName": DEPLOYMENT_NAME,
    },
    images={"primary_container_image": deployment_image},
    region_at_runtime=True,
)


@fl.workflow
def xgboost_fastapi_wf(
    execution_role_arn: str = os.getenv("EXECUTION_ROLE_ARN"),
    dataset: FlyteFile = "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv",
    deployment_name: str = "xgboost-fastapi",
    instance_type: str = "ml.m4.xlarge",
    initial_instance_count: int = 1,
    region: str = "us-east-2",
) -> list[dict]:
    serialized_model = train_model(dataset=dataset)
    compressed_model = convert_to_tar(model=serialized_model)
    return sagemaker_deployment_wf(
        deployment_name=deployment_name,
        model_path=compressed_model,
        execution_role_arn=execution_role_arn,
        instance_type=instance_type,
        initial_instance_count=initial_instance_count,
        region=region,
    )
