import os
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Request, Response, status
from xgboost import Booster, DMatrix


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
