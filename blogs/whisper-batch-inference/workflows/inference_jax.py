import functools
import json
import math
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import requests
from flax import jax_utils
from flax.core.frozen_dict import freeze
from flax.training.common_utils import shard
from flytekit import Resources, dynamic, map_task, task, workflow
from flytekit.types.file import FlyteFile
from transformers import WhisperProcessor, WhisperTokenizer
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from .modeling_flax_whisper import FlaxWhisperForConditionalGeneration


def chunk_iter_with_batch(
    inputs, chunk_len, stride_left, stride_right, batch_size, feature_extractor
):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right

    all_chunk_start_idx = np.arange(0, inputs_len, step)
    num_samples = len(all_chunk_start_idx)

    num_batches = math.ceil(num_samples / batch_size)
    batch_idx = np.array_split(np.arange(num_samples), num_batches)

    for _, idx in enumerate(batch_idx):
        chunk_start_idx = all_chunk_start_idx[idx]

        chunk_end_idx = chunk_start_idx + chunk_len

        chunks = [
            inputs[chunk_start:chunk_end]
            for chunk_start, chunk_end in zip(chunk_start_idx, chunk_end_idx)
        ]
        processed = feature_extractor(
            chunks,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="np",
        )

        _stride_left = np.where(chunk_start_idx == 0, 0, stride_left)
        is_last = np.where(
            stride_right > 0,
            chunk_end_idx > inputs_len,
            chunk_end_idx >= inputs_len,
        )
        _stride_right = np.where(is_last, 0, stride_right)

        chunk_lens = [chunk.shape[0] for chunk in chunks]
        strides = [
            [chunk_l, _stride_l.item(), _stride_r.item()]
            for chunk_l, _stride_l, _stride_r in zip(
                chunk_lens, _stride_left, _stride_right
            )
        ]

        yield {"stride": strides, **processed}


def preprocess_batch(
    inputs,
    feature_extractor,
    chunk_length_s=30.0,
    stride_length_s=None,
    batch_size=None,
):
    if inputs.startswith("http://") or inputs.startswith("https://"):
        # We need to actually check for a real protocol, otherwise it's impossible to use a local file
        # like http_huggingface_co.png
        inputs = requests.get(inputs).content
    else:
        with open(inputs, "rb") as f:
            inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, feature_extractor.sampling_rate)

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
        )

    if chunk_length_s:
        if stride_length_s is None:
            stride_length_s = chunk_length_s / 6

        if isinstance(stride_length_s, (int, float)):
            stride_length_s = [stride_length_s, stride_length_s]

        chunk_len = round(chunk_length_s * feature_extractor.sampling_rate)
        stride_left = round(stride_length_s[0] * feature_extractor.sampling_rate)
        stride_right = round(stride_length_s[1] * feature_extractor.sampling_rate)

        if chunk_len < stride_left + stride_right:
            raise ValueError("Chunk length must be superior to stride length")

        for item in chunk_iter_with_batch(
            inputs, chunk_len, stride_left, stride_right, batch_size, feature_extractor
        ):
            yield item
    else:
        processed = feature_extractor(
            inputs,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="np",
        )
        yield processed


def get_forced_decoder_ids(
    model, generation_config=None, task=None, language=None, return_timestamps=False
):
    if generation_config is None:
        generation_config = model.generation_config

    if hasattr(generation_config, "is_multilingual"):
        is_multilingual = generation_config.is_multilingual
    else:
        is_multilingual = None

    forced_decoder_ids = []

    if is_multilingual:
        if language is not None:
            language = language.lower()
            if language in generation_config.lang_to_id.keys():
                language_token = language
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            elif language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            else:
                if len(language) == 2:
                    # ISO 639-1 language code
                    acceptable_languages = list(TO_LANGUAGE_CODE.values())
                elif "<" in language or "|" in language or ">" in language:
                    # generation config language code
                    acceptable_languages = list(generation_config.lang_to_id.keys())
                else:
                    # language passed as a string
                    acceptable_languages = list(TO_LANGUAGE_CODE.keys())
                raise ValueError(
                    f"Unsupported language: {language}. Language should be one of:"
                    f" {acceptable_languages}."
                )
            forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))

        if task is not None:
            forced_decoder_ids.append((2, generation_config.task_to_id[task]))
        else:
            forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))

    if not return_timestamps:
        if (
            forced_decoder_ids
            and forced_decoder_ids[-1][0] != generation_config.no_timestamps_token_id
        ):
            idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
            forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

    return forced_decoder_ids


def forward_generate(
    input_features,
    model,
    max_length,
    params,
    language=None,
    task=None,
    return_timestamps=False,
):
    def generate(params, input_features, forced_decoder_ids, return_timestamps):
        output_ids = model.pipeline_generate(
            input_features,
            params=params,
            forced_decoder_ids=forced_decoder_ids,
            return_timestamps=return_timestamps,
            max_length=max_length,
        )
        return output_ids

    p_generate = jax.pmap(
        generate,
        "input_features",
        in_axes=(0, 0, None),
        out_axes=0,
        static_broadcasted_argnums=(3,),
    )
    forced_decoder_ids = get_forced_decoder_ids(
        language=language, task=task, return_timestamps=return_timestamps, model=model
    )
    output_ids = p_generate(
        freeze(params),
        shard(input_features),
        forced_decoder_ids,
        return_timestamps,
    ).sequences
    output_ids = jax.device_get(output_ids.reshape(-1, max_length))

    return output_ids


@task
def forward(
    model_inputs: List[np.ndarray],
    batch_size: Optional[int],
    language: Optional[str],
    task: Optional[str],
    return_timestamps: Optional[bool],
    max_length: Optional[int],
    checkpoint: str,
) -> List[np.ndarray]:
    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        checkpoint, _do_init=False, dtype=jnp.float16, cache_dir="whisper-models"
    )

    max_length = model.generation_config.max_length if max_length == 0 else max_length
    params = jax_utils.replicate(params)

    model_inputs = {
        "input_features": model_inputs[0],
        "stride": model_inputs[1].tolist(),
    }

    # We need to keep track of some additional input arguments for post-processing so need to forward these on after running generation
    input_features = model_inputs.pop("input_features")
    input_batch_size = input_features.shape[0]

    if input_batch_size != batch_size:
        padding = np.zeros(
            [batch_size - input_batch_size, *input_features.shape[1:]],
            input_features.dtype,
        )
        input_features = np.concatenate([input_features, padding])

    pred_ids = forward_generate(
        input_features=input_features,
        model=model,
        max_length=max_length,
        params=params,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
    )[:input_batch_size]

    # tokenizer's decode method expects an extra dim - we insert it here for convenience
    out = {"tokens": pred_ids[:, None, :]}

    stride = model_inputs.pop("stride", None)
    if stride is not None:
        out["stride"] = stride

    return [out["tokens"], np.array(out["stride"])]


@task(requests=Resources(mem="5Gi", cpu="2", gpu="1"))
def postprocess(
    model_outputs: List[List[np.ndarray]],
    chunk_length: int,
    sampling_rate: int,
    max_source_positions: int,
    tokenizer: WhisperTokenizer,
    return_timestamps: bool,
) -> str:
    unpacked_model_outputs = []
    for output in model_outputs:
        model_output = {"tokens": output[0], "stride": output[1].tolist()}
        for t in zip(*model_output.values()):
            unpacked_model_outputs.append(dict(zip(model_output, t)))

    time_precision = chunk_length / max_source_positions
    # Send the chunking back to seconds, it's easier to handle in whisper
    for output in unpacked_model_outputs:
        if "stride" in output:
            chunk_len, stride_left, stride_right = output["stride"]
            # Go back in seconds
            chunk_len /= sampling_rate
            stride_left /= sampling_rate
            stride_right /= sampling_rate
            output["stride"] = chunk_len, stride_left, stride_right

    text, optional = tokenizer._decode_asr(
        unpacked_model_outputs,
        return_timestamps=return_timestamps,
        return_language=None,
        time_precision=time_precision,
    )
    return json.dumps({"text": text, **optional})


@dynamic(requests=Resources(mem="10Gi", cpu="4", gpu="1"))
def jax_batch_inference(
    audios: List[FlyteFile],
    checkpoint: str,
    max_length: int,
    chunk_length_s: float,
    stride_length_s: float,
    batch_size: int,
    language: str,
    task: str,
    return_timestamps: bool,
) -> List[str]:
    processor = WhisperProcessor.from_pretrained(checkpoint)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    transcriptions = []
    for audio in audios:
        dataloader = preprocess_batch(
            inputs=audio.download(),
            feature_extractor=feature_extractor,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
        )

        dataloader_to_list = list(
            map(
                lambda batch: [batch["input_features"], np.array(batch["stride"])],
                dataloader,
            )
        )

        # iterate over our chunked audio samples
        map_task_partial = functools.partial(
            forward,
            batch_size=batch_size,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            max_length=max_length,
            checkpoint=checkpoint,
        )
        model_outputs = map_task(map_task_partial)(
            model_inputs=dataloader_to_list
        ).with_overrides(requests=Resources(mem="20Gi", cpu="2", gpu="1"))

        transcriptions.append(
            postprocess(
                model_outputs=model_outputs,
                chunk_length=feature_extractor.chunk_length,
                sampling_rate=feature_extractor.sampling_rate,
                # model.config.max_source_positions
                max_source_positions=1500,
                tokenizer=tokenizer,
                return_timestamps=return_timestamps,
            )
        )
    return transcriptions


@workflow
def jax_batch_inference_wf(
    checkpoint: str = "openai/whisper-large-v2",
    max_length: int = 0,
    chunk_length_s: float = 30.0,
    stride_length_s: float = 5.0,
    batch_size: int = 16,
    language: str = "en",
    task: str = "transcribe",
    return_timestamps: bool = False,
    audios: List[FlyteFile] = [
        "https://datasets-server.huggingface.co/assets/librispeech_asr/--/all/train.clean.100/1/audio/audio.mp3",
        "https://huggingface.co/datasets/Samhita/SadTalkerData/resolve/main/Audio%20-%20Oprah%20Winfrey.mp3",
        "https://datasets-server.huggingface.co/assets/sanchit-gandhi/whisper-jax-test-files/--/sanchit-gandhi--whisper-jax-test-files/train/0/audio/audio.mp3",
        "https://datasets-server.huggingface.co/assets/sanchit-gandhi/whisper-jax-test-files/--/sanchit-gandhi--whisper-jax-test-files/train/1/audio/audio.mp3",
        "https://huggingface.co/datasets/Samhita/whisper-jax-examples/resolve/main/khloe_kardashian_podcast.mp3",
    ],
) -> List[str]:
    return jax_batch_inference(
        checkpoint=checkpoint,
        max_length=max_length,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        batch_size=batch_size,
        language=language,
        task=task,
        return_timestamps=return_timestamps,
        audios=audios,
    )
