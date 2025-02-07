import functools
import os
from datetime import timedelta
from pathlib import Path

import flytekit as fl
import pandas as pd
from flytekit.types.file import FlyteFile
from pydantic import BaseModel
from union import ActorEnvironment

from .monologue_flow import DialogueEntry, PDFMetadata, image, monologue_workflow
from .podcast_flow import podcast_workflow
from .utils import eleven_key


class PDF(BaseModel):
    type: str
    pdf: FlyteFile


tts_actor = ActorEnvironment(
    name="tts",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=eleven_key)],
    container_image=image.with_packages(
        ["elevenlabs", "flytekitplugins-deck-standard"]
    ),
)


@fl.task(
    retries=3,
    container_image=image,
    requests=fl.Resources(mem="10Gi", cpu="5"),
    cache=True,
    cache_version="0.2",
)
def convert_pdf(pdf: PDF) -> PDFMetadata:
    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    result = converter.convert(pdf.pdf.download(), raises_on_error=True)
    file_path = str(result.input.file)

    status = (
        "success"
        if result.status in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}
        else "failed"
    )
    content = result.document.export_to_markdown() if status == "success" else ""
    error_msg = (
        "; ".join(str(error) for error in result.errors)
        if result.errors
        else f"Conversion failed with status: {result.status}"
    )

    return PDFMetadata(
        filename=os.path.basename(file_path),
        status=status,
        content=content,
        error=error_msg if status == "failed" else None,
        type=pdf.type,
    )


@tts_actor.task(enable_deck=True)
def get_available_voices() -> None:
    from elevenlabs.client import ElevenLabs
    from flytekitplugins.deck.renderer import TableRenderer

    eleven_labs_client = ElevenLabs(
        api_key=fl.current_context().secrets.get(key=eleven_key)
    )
    response = eleven_labs_client.voices.get_all()

    voices_data = []
    for voice in response.voices:
        voice_data = {
            "voice_id": voice.voice_id,
            "name": voice.name,
            "category": voice.category,
            "accent": voice.labels.get("accent"),
            "description": voice.labels.get("description"),
            "age": voice.labels.get("age"),
            "gender": voice.labels.get("gender"),
            "use_case": voice.labels.get("use_case"),
            "preview_url": voice.preview_url,
            "requires_verification": voice.voice_verification.requires_verification,
            "is_verified": voice.voice_verification.is_verified,
        }
        voices_data.append(voice_data)

    df = pd.DataFrame(voices_data)
    fl.Deck("Available Voices", TableRenderer().to_html(df=df))


@tts_actor.task
def convert_text_to_voice(
    dialogue_entry: DialogueEntry,
    speaker_names: dict[str, list[str]],
) -> bytes:
    from elevenlabs.client import ElevenLabs

    # Initialize ElevenLabs client
    eleven_labs_client = ElevenLabs(
        api_key=fl.current_context().secrets.get(key=eleven_key)
    )

    # Determine the voice based on the speaker
    speaker_index = 0 if dialogue_entry.speaker == "speaker-1" else 1
    voice = speaker_names["speaker_names"][speaker_index]

    # Generate audio stream
    audio_stream = eleven_labs_client.generate(
        text=dialogue_entry.text,
        voice=voice,
        model="eleven_monolingual_v1",
        voice_settings={"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
        stream=True,
    )

    # Combine audio chunks into a single byte stream
    return b"".join(audio_stream)


@tts_actor.task
def coalesce_dialogues(dialogues: list[bytes]) -> FlyteFile:
    working_dir = Path(fl.current_context().working_directory)
    podcast_file_path = working_dir / "podcast.mp3"

    with open(podcast_file_path, "wb") as file:
        for chunk in dialogues:
            file.write(chunk)

    return FlyteFile(path=podcast_file_path)


@fl.workflow
def pdf_to_podcast(
    pdfs: list[PDF] = [
        PDF(type="target", pdf=FlyteFile("samples/investorpres-main.pdf")),
        PDF(type="context", pdf=FlyteFile("samples/bofa-context.pdf")),
        PDF(type="context", pdf=FlyteFile("samples/citi-context.pdf")),
    ],
    is_monologue: bool = True,
) -> FlyteFile:
    pdf_metadata = fl.map_task(convert_pdf)(pdf=pdfs)

    available_voices = get_available_voices()
    speaker_names = fl.wait_for_input(
        name="speaker-names-list",
        timeout=timedelta(minutes=15),
        expected_type=list[str],
    )
    available_voices >> speaker_names

    conversation = (
        fl.conditional("pdf-to-podcast")
        .if_(is_monologue.is_true())
        .then(monologue_workflow(pdf_metadata, speaker_names[0]))
        .else_()
        .then(podcast_workflow(pdf_metadata, speaker_names[0], speaker_names[1]))
    )

    dialogues = fl.map_task(
        functools.partial(
            convert_text_to_voice,
            speaker_names={
                "speaker_names": speaker_names
            },  # cannot send a list to partial
        )
    )(dialogue_entry=conversation.dialogue)

    return coalesce_dialogues(dialogues)
