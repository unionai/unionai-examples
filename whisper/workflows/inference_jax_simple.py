import json

import jax.numpy as jnp
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile
from whisper_jax import FlaxWhisperPipline


@task(
    requests=Resources(gpu="1", mem="15Gi", cpu="2"),
)
def jax_transcribe(
    audio: FlyteFile,
    chunk_length_s: float,
    stride_length_s: float,
    batch_size: int,
    language: str,
    task: str,
    return_timestamps: bool,
    checkpoint: str,
) -> str:
    pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.float16, batch_size=batch_size)
    return json.dumps(
        pipeline(
            audio.download(),
            chunk_length_s,
            stride_length_s,
            batch_size,
            language,
            task,
            return_timestamps,
        )
    )


@workflow
def jax_simple_wf(
    audio: FlyteFile = "https://huggingface.co/datasets/Samhita/whisper-jax-examples/resolve/main/khloe_kardashian_podcast.mp3",
    checkpoint: str = "openai/whisper-large-v2",
    chunk_length_s: float = 30.0,
    stride_length_s: float = 5.0,
    batch_size: int = 16,
    language: str = "en",
    task: str = "transcribe",
    return_timestamps: bool = False,
) -> str:
    return jax_transcribe(
        audio=audio,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        batch_size=batch_size,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        checkpoint=checkpoint,
    )
