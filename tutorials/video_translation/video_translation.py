# # Video Dubbing
#
# This tutorial demonstrates a video dubbing workflow using open-source models.
#
# The workflow consists of the following steps:
# 1. Extract audio and image from the video.
# 2. Transcribe the audio to text using Whisper.
# 3. Translate the text to the desired language using M2M100.
# 4. Clone the original speaker's voice using Coqui TTS.
# 5. Lip sync the cloned voice to the video using SadTalker.

# Start by importing the necessary libraries and modules:

import os
import shutil
from pathlib import Path
from typing import Optional

import flytekit
import numpy as np
import requests
from flytekit import Resources, task, workflow
from flytekit.extras.accelerators import T4
from flytekit.types.file import FlyteFile

from utils import (
    AudioAndImageValues,
    clone_voice_image,
    clone_voice_language_codes,
    language_codes,
    language_translation_image,
    lip_sync_image,
    preprocessing_image,
    speech2text_image,
)


# ## Audio & image extraction
#
# This task extracts audio from the video file and selects a representative frame for lip syncing.
# The [Katna](https://github.com/keplerlab/katna) library is used to choose the most representative keyframe.

@task(
    cache=True,
    cache_version="2",
    container_image=preprocessing_image,
    requests=Resources(mem="5Gi", cpu="1"),
    accelerator=T4,
)
def fetch_audio_and_image(
    video_file: FlyteFile, output_ext: str
) -> AudioAndImageValues:
    from Katna.video import Video
    from Katna.writer import KeyFrameDiskWriter
    from moviepy.editor import VideoFileClip

    downloaded_video = video_file.download()

    # AUDIO
    video_filename, _ = os.path.splitext(downloaded_video)
    clip = VideoFileClip(downloaded_video)

    audio_file_path = Path(
        flytekit.current_context().working_directory, f"{video_filename}.{output_ext}"
    ).as_posix()
    clip.audio.write_audiofile(audio_file_path)

    # IMAGE
    if os.path.splitext(downloaded_video)[1] == "":
        new_file_name = downloaded_video + ".mp4"
        os.rename(downloaded_video, new_file_name)
        downloaded_video = new_file_name

    image_dir = flytekit.current_context().working_directory

    vd = Video()
    no_of_frames_to_return = 1

    # initialize diskwriter to save data at desired location
    diskwriter = KeyFrameDiskWriter(location=image_dir)
    print(f"Input video file path = {downloaded_video}")

    # extract the best keyframe and process data with diskwriter
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_return,
        file_path=downloaded_video,
        writer=diskwriter,
    )

    return AudioAndImageValues(
        audio=FlyteFile(audio_file_path),
        image=FlyteFile(Path(image_dir, os.listdir(image_dir)[0]).as_posix()),
    )


# ## Speech-to-text transcription
#
# This task transcribes the extracted audio to text using the [Whisper](https://huggingface.co/openai/whisper-large-v2) model.
# The transcription enables translation in the subsequent task.

@task(
    cache=True,
    cache_version="2",
    container_image=speech2text_image,
    requests=Resources(gpu="1", mem="10Gi", cpu="1"),
    accelerator=T4,
    environment={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"},
)
def speech2text(
    checkpoint: str,
    audio: FlyteFile,
    chunk_length: float,
    return_timestamps: bool,
    translate_from: str,
) -> str:
    import torch
    from transformers import pipeline
    from transformers.pipelines.audio_utils import ffmpeg_read

    pipe = pipeline(
        "automatic-speech-recognition",
        model=checkpoint,
        chunk_length_s=chunk_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    local_audio_path = audio.download()
    if local_audio_path.startswith("http://") or local_audio_path.startswith(
        "https://"
    ):
        inputs = requests.get(inputs).content
    else:
        with open(local_audio_path, "rb") as f:
            inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")

    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
        )

    prediction = pipe(
        inputs,
        return_timestamps=return_timestamps,
        generate_kwargs={"task": "transcribe", "language": translate_from},
    )
    output = prediction["text"].strip()
    return output


# The `checkpoint` parameter specifies the speech-to-text model to use.
# This task uses a T4 GPU accelerator for faster execution.
# The accelerator type can be modified in the `@task` decorator to use a different GPU if needed.


# ## Text translation
#
# This task translates the transcribed text to the desired language using the [M2M100](https://huggingface.co/facebook/m2m100_1.2B)
# model, and doesn't require a GPU.
# Both source and target languages are provided as inputs when executing the workflow.

@task(
    cache=True,
    cache_version="2",
    container_image=language_translation_image,
    requests=Resources(mem="10Gi", cpu="3"),
)
def translate_text(translate_from: str, translate_to: str, input: str) -> str:
    import nltk
    from nltk import sent_tokenize
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    if translate_to not in clone_voice_language_codes:
        raise ValueError(f"{translate_to} language isn't supported by Coqui TTS model.")

    if translate_to not in language_codes:
        raise ValueError(f"{translate_to} language isn't supported by M2M100 model.")

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")

    tokenizer.src_lang = language_codes[translate_from]

    nltk.download("punkt")
    result = []
    for sentence in sent_tokenize(input):
        encoded_input = tokenizer(sentence, return_tensors="pt")

        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.get_lang_id(language_codes[translate_to]),
        )
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        result += output

    return " ".join(result).strip()


# ## Voice cloning
#
# This task uses [Coqui XTTS](https://huggingface.co/coqui/XTTS-v2) to generate speech in the target language while preserving
# the original speaker's voice characteristics.

@task(
    cache=True,
    cache_version="2",
    container_image=clone_voice_image,
    requests=Resources(gpu="1", mem="15Gi"),
    accelerator=T4,
    environment={"COQUI_TOS_AGREED": "1"},
)
def clone_voice(text: str, target_lang: str, speaker_wav: FlyteFile) -> FlyteFile:
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    file_path = Path(
        flytekit.current_context().working_directory, "output.wav"
    ).as_posix()

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav.download(),
        language=clone_voice_language_codes[target_lang],
        file_path=file_path,
        split_sentences=True,
    )
    return FlyteFile(file_path)


# ## Lip sync
#
# This task uses [SadTalker](https://github.com/OpenTalker/SadTalker) to synchronize the audio with the video.
# The model allows for adjusting various parameters such as pose style, face enhancement,
# background enhancement, and expression scale.

@task(
    cache=True,
    cache_version="2",
    requests=(Resources(gpu="1", mem="30Gi")),
    container_image=lip_sync_image,
    accelerator=T4,
)
def lip_sync(
    audio_path: FlyteFile,
    pic_path: FlyteFile,
    ref_pose: FlyteFile,
    ref_eyeblink: FlyteFile,
    pose_style: int,
    batch_size: int,
    expression_scale: float,
    input_yaw_list: Optional[list[int]],
    input_pitch_list: Optional[list[int]],
    input_roll_list: Optional[list[int]],
    enhancer: str,
    background_enhancer: str,
    device: str,
    still: bool,
    preprocess: str,
    checkpoint_dir: str,
    size: int,
) -> FlyteFile:
    from lip_sync_src.facerender.animate import AnimateFromCoeff
    from lip_sync_src.generate_batch import get_data
    from lip_sync_src.generate_facerender_batch import get_facerender_data
    from lip_sync_src.test_audio2coeff import Audio2Coeff
    from lip_sync_src.utils.init_path import init_path
    from lip_sync_src.utils.preprocess import CropAndExtract

    audio_path = audio_path.download()
    pic_path = pic_path.download()

    if ref_eyeblink.remote_source == ref_pose.remote_source:
        ref_eyeblink = ref_eyeblink.download()
        ref_pose = ref_eyeblink
    else:
        ref_eyeblink = ref_eyeblink.download()
        ref_pose = ref_pose.download()

    working_dir = flytekit.current_context().working_directory
    save_dir = os.path.join(working_dir, "result")
    os.makedirs(save_dir, exist_ok=True)

    sadtalker_paths = init_path(
        checkpoint_dir,
        os.path.join("/root", "lip_sync_src", "config"),
        size,
        preprocess,
    )

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)

    print("3DMM Extraction for source image")
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path,
        first_frame_dir,
        preprocess,
        source_image_flag=True,
        pic_size=size,
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink != "":
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print("3DMM Extraction for the reference video providing eye blinking")
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(
            ref_eyeblink,
            ref_eyeblink_frame_dir,
            preprocess,
            source_image_flag=False,
        )
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose != "":
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print("3DMM Extraction for the reference video providing pose")
            ref_pose_coeff_path, _, _ = preprocess_model.generate(
                ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False
            )
    else:
        ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(
        first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still
    )
    coeff_path = audio_to_coeff.generate(
        batch, save_dir, pose_style, ref_pose_coeff_path
    )

    # coeff2video
    data = get_facerender_data(
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        input_yaw_list,
        input_pitch_list,
        input_roll_list,
        expression_scale=expression_scale,
        still_mode=still,
        preprocess=preprocess,
        size=size,
    )

    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        preprocess=preprocess,
        img_size=size,
    )

    file_path = Path(save_dir, "output.mp4").as_posix()

    shutil.move(result, file_path)
    print("The generated video is named: ", file_path)

    return FlyteFile(file_path)


# Finally, wrap all the tasks into a workflow.

@workflow
def video_translation_wf(
    video_file: FlyteFile = "https://github.com/samhita-alla/video-translation/assets/27777173/d756f94e-54b5-43eb-a546-8f141e828ce2",
    translate_from: str = "English",
    translate_to: str = "German",
    checkpoint: str = "openai/whisper-large-v2",
    output_ext: str = "mp3",
    chunk_length: float = 30.0,
    return_timestamps: bool = False,
    ref_pose: FlyteFile = "https://github.com/Zz-ww/SadTalker-Video-Lip-Sync/raw/master/sync_show/original.mp4",
    ref_eyeblink: FlyteFile = "https://github.com/Zz-ww/SadTalker-Video-Lip-Sync/raw/master/sync_show/original.mp4",
    pose_style: int = 0,
    batch_size: int = 2,
    expression_scale: float = 1.0,
    input_yaw_list: Optional[list[int]] = None,
    input_pitch_list: Optional[list[int]] = None,
    input_roll_list: Optional[list[int]] = None,
    enhancer: str = "gfpgan",
    background_enhancer: str = "",
    device: str = "cuda",
    still: bool = True,
    preprocess: str = "extfull",
    size: int = 512,
    checkpoint_dir: str = "vinthony/SadTalker-V002rc",  # HF model
) -> FlyteFile:
    """
    Video translation Flyte workflow.

    :param video_file: The video file to translate.
    :param translate_from: The language to translate from.
    :param translate_to: The language to translate to, options are ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Polish', 'Turkish', 'Russian', 'Dutch', 'Czech', 'Arabic', 'Chinese', 'Japanese', 'Hungarian', 'Korean', 'Hindi']
    :param checkpoint: Speech-to-text model checkpoint.
    :param output_ext: Output extension for audio files.
    :param chunk_length: Length of audio chunks.
    :param return_timestamps: If set to True, provides start and end timestamps for each recognized word or segment in the output, along with the transcribed text.
    :param ref_pose: Path to reference video providing pose.
    :param ref_eyeblink: Path to reference video providing eye blinking.
    :param pose_style: Input pose style from [0, 46).
    :param batch_size: Batch size of facerender.
    :param expression_scale: A larger value will make the expression motion stronger.
    :param input_yaw_list: The input yaw degree of the user.
    :param input_pitch_list: The input pitch degree of the user.
    :param input_roll_list: The input roll degree of the user.
    :param enhancer: Face enhancer options include [gfpgan, RestoreFormer].
    :param background_enhancer: Background enhancer options include [realesrgan].
    :param device: The device to use, CPU or GPU.
    :param still: Can crop back to the original videos for the full body animation.
    :param preprocess: How to preprocess the images, options are ['crop', 'extcrop', 'resize', 'full', 'extfull'].
    :param size: The image size of the facerender, options are [256, 512].
    :param checkpoint_dir: Path to model checkpoint, currently hosted in a Hugging Face repository.
    """
    values = fetch_audio_and_image(video_file=video_file, output_ext=output_ext)
    text = speech2text(
        checkpoint=checkpoint,
        audio=values.audio,
        chunk_length=chunk_length,
        return_timestamps=return_timestamps,
        translate_from=translate_from,
    )
    translated_text = translate_text(
        translate_from=translate_from, translate_to=translate_to, input=text
    )
    cloned_voice = clone_voice(
        text=translated_text, target_lang=translate_to, speaker_wav=values.audio
    )
    return lip_sync(
        audio_path=cloned_voice,
        pic_path=values.image,
        ref_pose=ref_pose,
        ref_eyeblink=ref_eyeblink,
        pose_style=pose_style,
        batch_size=batch_size,
        expression_scale=expression_scale,
        input_yaw_list=input_yaw_list,
        input_pitch_list=input_pitch_list,
        input_roll_list=input_roll_list,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        device=device,
        still=still,
        preprocess=preprocess,
        size=size,
        checkpoint_dir=checkpoint_dir,
    )


# To run the workflow on Union, use the following command:
#
# ```
# union run --remote --copy-all video_translation.py video_translation_wf
# ```
