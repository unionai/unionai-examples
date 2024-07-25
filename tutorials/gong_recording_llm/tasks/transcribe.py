from flytekit import Resources, task, ImageSpec
from flytekit.extras.accelerators import T4


import json
import os
import numpy as np
import requests
import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

from tasks.download_call import CallData


torch_transcribe_img = ImageSpec(
    packages=[
        "torch==2.3.1",
        "transformers==4.42.4",
    ],
    apt_packages=["ffmpeg"],
    cuda="11.2.2",
    cudnn="8",
    env={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"},
)


@task(
    container_image=torch_transcribe_img,
    requests=Resources(gpu="1", mem="10Gi", cpu="1"),
    accelerator=T4,
    cache=True,
    cache_version="1.0"
)
def torch_transcribe(audio: CallData) -> str:

    checkpoint = "openai/whisper-large-v2"
    chunk_length = 30.0
    batch_size = 8
    return_timestamps = False

    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        chunk_length_s=chunk_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    local_audio_path = audio.call_audio.download()

    with open(local_audio_path, "rb") as f:
        input_features = f.read()

    if isinstance(input_features, bytes):
        input_features = ffmpeg_read(input_features, 16000)

    if not isinstance(input_features, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(input_features)}`")

    if len(input_features.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
        )

    prediction = pipe(
        input_features, batch_size=batch_size, return_timestamps=return_timestamps
    )
    return json.dumps(prediction)