import os
import tarfile
from contextlib import asynccontextmanager
from datetime import datetime

import flytekit
import numpy as np
from fastapi import FastAPI, Request, Response, status
from flytekit import ImageSpec, task, workflow
from flytekit.types.file import FlyteFile
from numpy import loadtxt
from sklearn.model_selection import train_test_split

custom_image = ImageSpec(
    name="sagemaker-xgboost",
    registry="<YOUR-REGISTRY>",
    requirements="requirements.txt",
    apt_packages=["git"],
    source_root=".",
).with_commands(["chmod +x /root/serve"])

if custom_image.is_container():
    from xgboost import Booster, DMatrix, XGBClassifier


########################
# MODEL TRAINING START #
########################


@task(container_image=custom_image)
def train_model(dataset: FlyteFile) -> FlyteFile:
    dataset = loadtxt(dataset.download(), delimiter=",")
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.33, random_state=7)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    serialized_model = os.path.join(
        flytekit.current_context().working_directory, "xgboost_model.json"
    )
    booster = model.get_booster()
    booster.save_model(serialized_model)

    return FlyteFile(path=serialized_model)


@task
def convert_to_tar(model: FlyteFile) -> FlyteFile:
    tf = tarfile.open("model.tar.gz", "w:gz")
    tf.add(model.download(), arcname="xgboost_model")
    tf.close()

    return FlyteFile("model.tar.gz")


@workflow
def sagemaker_xgboost_wf(
    dataset: FlyteFile = "https://gist.githubusercontent.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f/raw/819b69b5736821ccee93d05b51de0510bea00294/pima-indians-diabetes.csv",
) -> FlyteFile:
    serialized_model = train_model(dataset=dataset)
    return convert_to_tar(model=serialized_model)


######################
# MODEL TRAINING END #
######################

#####################
# FASTAPI APP START #
#####################


class Predictor:
    def __init__(self, path: str, name: str):
        self._model = Booster()
        self._model.load_model(os.path.join(path, name))

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


###################
# FASTAPI APP END #
###################
