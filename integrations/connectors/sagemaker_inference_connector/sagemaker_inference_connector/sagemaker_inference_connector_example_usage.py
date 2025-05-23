# # Deploy and serve an XGBoost model on AWS SageMaker using FastAPI
#
# {{run-on-union}}
#
# This example demonstrates how to deploy and serve an XGBoost model on SageMaker using FastAPI custom inference.
#
# The model artifact needs to be available in an S3 bucket for SageMaker to be able to access.
# We train an XGBoost model on the Pima Indians Diabetes dataset and generate a `tar.gz` file to be stored in an S3 bucket.
import os
import tarfile
from pathlib import Path

import union
from numpy import loadtxt
from sklearn.model_selection import train_test_split

train_model_image = union.ImageSpec(
    name="xgboost-train",
    registry="ghcr.io/flyteorg",
    packages=["xgboost"],
)

if train_model_image.is_container():
    from xgboost import XGBClassifier


@union.task(container_image=train_model_image)
def train_model(dataset: union.FlyteFile) -> union.FlyteFile:
    dataset = loadtxt(dataset.download(), delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=7)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    serialized_model = str(Path(union.current_context().working_directory) / "xgboost_model.json")
    booster = model.get_booster()
    booster.save_model(serialized_model)

    return union.FlyteFile(path=serialized_model)


@union.task
def convert_to_tar(model: union.FlyteFile) -> union.FlyteFile:
    tf = tarfile.open("model.tar.gz", "w:gz")
    tf.add(model.download(), arcname="xgboost_model")
    tf.close()

    return union.FlyteFile("model.tar.gz")


@union.workflow
def sagemaker_xgboost_wf(
    dataset: union.FlyteFile = "https://dub.sh/VZrumbQ",
) -> union.FlyteFile:
    serialized_model = train_model(dataset=dataset)
    return convert_to_tar(model=serialized_model)


# > [!NOTE]
# > Replace `ghcr.io/flyteorg` with a container registry to which you can publish.
# > To upload the image to the local registry in the demo cluster, indicate the registry as `localhost:30000`.
#
# The above workflow generates a compressed model artifact that can be stored in an S3 bucket.
# Take note of the S3 URI.
#
# To deploy the model on SageMaker, use the `flytekitplugins.awssagemaker_inference.create_sagemaker_deployment` function.
from flytekit import kwtypes
from flytekitplugins.awssagemaker_inference import create_sagemaker_deployment

REGION = "us-east-2"
S3_OUTPUT_PATH = "s3://sagemaker-agent-xgboost/inference-output/output"
DEPLOYMENT_NAME = "xgboost-fastapi"

sagemaker_image = union.ImageSpec(
    name="sagemaker-xgboost",
    registry="ghcr.io/flyteorg",  # Amazon EC2 Container Registry or a Docker registry accessible from your VPC.
    packages=["xgboost", "fastapi", "uvicorn", "scikit-learn"],
    source_root=".",
).with_commands(["chmod +x /root/serve"])


sagemaker_deployment_wf = create_sagemaker_deployment(
    name="xgboost-fastapi",
    model_input_types=kwtypes(model_path=str, execution_role_arn=str),
    model_config={
        "ModelName": DEPLOYMENT_NAME,
        "PrimaryContainer": {
            "Image": "{images.primary_container_image}",
            "ModelDataUrl": "{inputs.model_path}",
        },
        "ExecutionRoleArn": "{inputs.execution_role_arn}",
    },
    endpoint_config_input_types=kwtypes(instance_type=str),
    endpoint_config_config={
        "EndpointConfigName": DEPLOYMENT_NAME,
        "ProductionVariants": [
            {
                "VariantName": "variant-name-1",
                "ModelName": DEPLOYMENT_NAME,
                "InitialInstanceCount": 1,
                "InstanceType": "{inputs.instance_type}",
            },
        ],
        "AsyncInferenceConfig": {"OutputConfig": {"S3OutputPath": S3_OUTPUT_PATH}},
    },
    endpoint_config={
        "EndpointName": DEPLOYMENT_NAME,
        "EndpointConfigName": DEPLOYMENT_NAME,
    },
    images={"primary_container_image": sagemaker_image},
    region=REGION,
    idempotence_token=True,  # set to True by default
)


# This function returns an imperative workflow responsible for deploying the XGBoost model, creating an endpoint configuration
# and initializing an endpoint. Configurations relevant to these tasks are passed to the
# `flytekitplugins.awssagemaker_inference.create_sagemaker_deployment` function.
#
# An idempotence token ensures the generation of unique tokens for each configuration, preventing name collisions during updates.
# By default, `idempotence_token` in `create_sagemaker_deployment` is set to `True`, causing the connector to append an idempotence token to the
# model name, endpoint config name, and endpoint.
#
# - If a field value isn't provided (e.g., `ModelName`), the connector appends the idempotence token to the workflow name and uses that as the `ModelName`.
# - You can also manually set the idempotence token by adding `{idempotence_token}` to the relevant fields in the configuration, e.g., `xgboost-{idempotence_token}`.
#
# `sagemaker_image` should include the inference code, necessary libraries, and an entrypoint for model serving.
#
# > [!NOTE]
# > For more detailed instructions on using your custom inference image, refer to the
# > [Amazon SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).
#
# If the plugin attempts to create a deployment that already exists, it will return the existing ARNs instead of raising an error.
#
# > [!NOTE]
# > When two executions run in parallel and attempt to create the same endpoint, one execution will proceed with creating the endpoint while both will wait until the endpoint creation process is complete.
#
# To receive inference requests, the container built with `sagemaker_image` must have a web server
# listening on port 8080 and must accept POST and GET requests to the `/invocations` and `/ping` endpoints, respectively.
#
# We define the FastAPI inference code as follows:
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Request, Response, status

if sagemaker_image.is_container():
    from xgboost import Booster, DMatrix


class Predictor:
    def __init__(self, path: str, name: str):
        self._model = Booster()
        self._model.load_model(str(Path(path) / name))

    def predict(self, inputs: DMatrix) -> np.ndarray:
        return self._model.predict(inputs)


ml_model: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    path = os.getenv("MODEL_PATH", "/opt/ml/model")
    ml_model = Predictor(path=path, name="xgboost_model")
    yield
    ml_model = None


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return Response(content="OK", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    print(f"Received request at {datetime.now()}")

    json_payload = await request.json()

    X_test = DMatrix(np.array(json_payload).reshape((1, -1)))
    y_test = ml_model.predict(X_test)

    response = Response(
        content=repr(round(y_test[0])).encode("utf-8"),
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response


# Create a file named `serve` to serve the model. In our case, we are using FastAPI:
#
# ```shell
# !/bin/bash
#
# _term() {
# echo "Caught SIGTERM signal!"
# kill -TERM "$child" 2>/dev/null
# }
#
# trap _term SIGTERM
#
# echo "Starting the API server"
# uvicorn sagemaker_inference_agent_example_usage:app --host 0.0.0.0 --port 8080&
#
# child=$!
# wait "$child"
# ```
#
# You can trigger the `sagemaker_deployment_wf` by providing the model artifact path,
# execution role ARN, and instance type.
#
# Once the endpoint creation status changes to `InService`, the SageMaker deployment workflow succeeds.
# You can then invoke the endpoint using the SageMaker connector as follows:
from flytekitplugins.awssagemaker_inference import SageMakerInvokeEndpointTask

invoke_endpoint = SageMakerInvokeEndpointTask(
    name="sagemaker_invoke_endpoint",
    config={
        "EndpointName": "YOUR_ENDPOINT_NAME_HERE",
        "InputLocation": "s3://sagemaker-agent-xgboost/inference_input",
    },
    region=REGION,
)

# The `flytekitplugins.awssagemaker_inference.SageMakerInvokeEndpointTask` invokes an endpoint asynchronously, resulting in an
# S3 location that will be populated with the output after it's generated.
# For instance, the inference_input file may include input like this: `[6, 148, 72, 35, 0, 33.6, 0.627, 50]`
#
# To delete the deployment, you can instantiate a `flytekitplugins.awssagemaker_inference.delete_sagemaker_deployment` function.
from flytekitplugins.awssagemaker_inference import delete_sagemaker_deployment

sagemaker_deployment_deletion_wf = delete_sagemaker_deployment(name="sagemaker-deployment-deletion", region="us-east-2")


@workflow
def deployment_deletion_workflow():
    sagemaker_deployment_deletion_wf(
        endpoint_name="YOUR_ENDPOINT_NAME_HERE",
        endpoint_config_name="YOUR_ENDPOINT_CONFIG_NAME_HERE",
        model_name="YOUR_MODEL_NAME_HERE",
    )


# You need to provide the endpoint name, endpoint config name, and the model name
# to execute this deletion, which removes the endpoint, endpoint config, and the model.
#
# ## Available tasks
#
# You have the option to execute the SageMaker tasks independently. The following tasks are available for use:
#
# - `flytekitplugins.awssagemaker_inference.SageMakerModelTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerEndpointConfigTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerEndpointTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerDeleteEndpointTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerDeleteEndpointConfigTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerDeleteModelTask`
# - `flytekitplugins.awssagemaker_inference.SageMakerInvokeEndpointTask`
#
# All tasks except the `flytekitplugins.awssagemaker_inference.SageMakerEndpointTask`
# inherit the `flytekitplugins.awssagemaker_inference.BotoTask`.
# The `flytekitplugins.awssagemaker_inference.BotoTask` provides the flexibility to invoke any
# [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) method.
# If you need to interact with the Boto3 APIs, you can use this task.
