import base64
import io
import os
from contextlib import asynccontextmanager
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from PIL import Image
from transformers import SamProcessor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class Predictor:
    def __init__(self, path: str, name: str):
        if torch.cuda.is_available():
            map_location = "cuda:0"
        else:
            map_location = torch.device("cpu")
        self._model = torch.load(os.path.join(path, name), map_location=map_location)

    def predict(self, input: dict) -> np.ndarray:
        # prepare image + box prompt for the model
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        image = Image.open(
            io.BytesIO(base64.b64decode(input["image_data"].encode("utf-8")))
        )
        inputs = processor(
            image, input_boxes=[[input["prompt"]]], return_tensors="pt"
        ).to(device)

        self._model.eval()

        # forward pass
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)

        # apply sigmoid
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))

        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        fig, axes = plt.subplots()
        axes.imshow(np.array(image))
        show_mask(medsam_seg, axes)

        file_path = "predicted_image.png"

        # Save the plot as an image
        plt.savefig(file_path)

        # Close the plot to release memory
        plt.close(fig)

        return file_path


sam_model: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam_model
    path = os.getenv("MODEL_PATH", "/opt/ml/model")
    sam_model = Predictor(path=path, name="sam_finetuned")
    yield
    sam_model = None


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return Response(content="OK", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    print(f"Received request at {datetime.now()}")

    json_payload = await request.json()
    file_path = sam_model.predict(json_payload)

    return FileResponse(file_path)
