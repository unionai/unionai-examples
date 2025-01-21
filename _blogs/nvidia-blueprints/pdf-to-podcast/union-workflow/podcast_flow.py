import functools
from typing import Optional

import flytekit as fl
from pydantic import BaseModel
from union.actor import ActorEnvironment

from .monologue_flow import DEFAULT_CONFIGS, Conversation, PDFMetadata
from .podcast_prompts import PodcastPrompts
from .utils import image, nvidia_key

podcast_actor = ActorEnvironment(
    name="podcast-flow-actor",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=nvidia_key)],
    container_image=image,
    requests=fl.Resources(mem="10Gi", cpu="5"),
)


class SegmentPoint(BaseModel):
    description: str


class SegmentTopic(BaseModel):
    title: str
    points: list[SegmentPoint]


class PodcastSegment(BaseModel):
    section: str
    topics: list[SegmentTopic]
    duration: int
    references: list[str]


class PodcastOutline(BaseModel):
    title: str
    segments: list[PodcastSegment]


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_summarize_pdf(pdf_metadata: PDFMetadata, retries: int) -> PDFMetadata:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    template = PodcastPrompts.get_template("podcast_summary_prompt")

    prompt = template.render(text=pdf_metadata.content)
    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["reasoning"]["name"],
        base_url=DEFAULT_CONFIGS["reasoning"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    summary_response = llm.invoke(
        [{"role": "user", "content": prompt}],
    )
    pdf_metadata.summary = summary_response.content

    return pdf_metadata


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_generate_raw_outline(
    summarized_pdfs: list[PDFMetadata],
    duration: int,
    guide_prompt: Optional[str] = None,
    retries: int = 5,
) -> str:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Prepare document summaries in XML format
    documents = []
    for pdf in summarized_pdfs:
        doc_str = f"""
        <document>
        <type>{"Target Document" if pdf.type == "target" else "Context Document"}</type>
        <path>{pdf.filename}</path>
        <summary>
        {pdf.summary}
        </summary>
        </document>"""
        documents.append(doc_str)

    template = PodcastPrompts.get_template("podcast_multi_pdf_outline_prompt")
    prompt = template.render(
        total_duration=duration,
        focus_instructions=guide_prompt,
        documents="\n\n".join(documents),
    )

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["reasoning"]["name"],
        base_url=DEFAULT_CONFIGS["reasoning"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    raw_outline = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    return raw_outline.content


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_generate_structured_outline(
    raw_outline: str, pdf_metadata: list[PDFMetadata], retries: int = 5
) -> PodcastOutline:
    import ujson as json
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Force the model to only reference valid filenames
    valid_filenames = [pdf.filename for pdf in pdf_metadata]
    schema = PodcastOutline.model_json_schema()
    schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
        "type": "string",
        "enum": valid_filenames,
    }

    template = PodcastPrompts.get_template(
        "podcast_multi_pdf_structured_outline_prompt"
    )
    prompt = template.render(
        outline=raw_outline,
        schema=json.dumps(schema, indent=2),
        valid_filenames=[pdf.filename for pdf in pdf_metadata],
    )

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["json"]["name"],
        base_url=DEFAULT_CONFIGS["json"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_structured_output(schema)
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    outline = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    return PodcastOutline.model_validate(outline)


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_process_segments(
    outline_segment: PodcastSegment,
    pdf_metadata: dict[str, list[PDFMetadata]],
    retries: int,
) -> str:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Get reference content if it exists
    text_content = []
    if outline_segment.references:
        for ref in outline_segment.references:
            # Find matching PDF metadata by filename
            pdf = next(
                (pdf for pdf in pdf_metadata["pdf_metadata"] if pdf.filename == ref),
                None,
            )
            if pdf:
                text_content.append(pdf.content)

    # Choose template based on whether we have references
    template_name = (
        "podcast_prompt_with_references"
        if text_content
        else "podcast_prompt_no_references"
    )
    template = PodcastPrompts.get_template(template_name)

    # Prepare prompt parameters
    prompt_params = {
        "duration": outline_segment.duration,
        "topic": outline_segment.section,
        "angles": "\n".join([topic.title for topic in outline_segment.topics]),
    }

    # Add text content if we have references
    if text_content:
        prompt_params["text"] = "\n\n".join(text_content)

    prompt = template.render(**prompt_params)

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["iteration"]["name"],
        base_url=DEFAULT_CONFIGS["iteration"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    response = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    return response.content


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_generate_dialogue(
    outline_segment: PodcastSegment,
    segment: str,
    speaker_1_name: str,
    speaker_2_name: str,
    retries: int,
) -> dict[str, str]:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    topics_text = "\n".join(
        [
            f"- {topic.title}\n"
            + "\n".join([f"  * {point.description}" for point in topic.points])
            for topic in outline_segment.topics
        ]
    )

    # Generate dialogue using template
    template = PodcastPrompts.get_template("podcast_transcript_to_dialogue_prompt")
    prompt = template.render(
        text=segment,
        duration=outline_segment.duration,
        descriptions=topics_text,
        speaker_1_name=speaker_1_name,
        speaker_2_name=speaker_2_name,
    )

    # Query LLM for dialogue
    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["reasoning"]["name"],
        base_url=DEFAULT_CONFIGS["reasoning"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    dialogue_response = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    return {"section": outline_segment.section, "dialogue": dialogue_response.content}


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_combine_dialogues(
    segment_dialogues: list[dict[str, str]], outline: PodcastOutline, retries: int
) -> str:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Start with the first segment's dialogue
    current_dialogue = segment_dialogues[0]["dialogue"]

    # Iteratively combine with subsequent segments
    for idx in range(1, len(segment_dialogues)):
        next_section = segment_dialogues[idx]["dialogue"]
        current_section = segment_dialogues[idx]["section"]

        template = PodcastPrompts.get_template("podcast_combine_dialogues_prompt")
        prompt = template.render(
            outline=outline.model_dump_json(),
            dialogue_transcript=current_dialogue,
            next_section=next_section,
            current_section=current_section,
        )

        llm = ChatNVIDIA(
            model=DEFAULT_CONFIGS["iteration"]["name"],
            base_url=DEFAULT_CONFIGS["iteration"]["api_base"],
            nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
            max_tokens=None,
        )
        llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
        combined = llm.invoke(
            [{"role": "user", "content": prompt}],
        )

        current_dialogue = combined.content

    return current_dialogue


@podcast_actor.task(cache=True, cache_version="0.2")
def podcast_create_final_conversation(
    dialogue: str, speaker_1_name: str, speaker_2_name: str, retries: int
) -> Conversation:
    import ujson as json
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    schema = Conversation.model_json_schema()
    template = PodcastPrompts.get_template("podcast_dialogue_prompt")
    prompt = template.render(
        speaker_1_name=speaker_1_name,
        speaker_2_name=speaker_2_name,
        text=dialogue,
        schema=json.dumps(schema, indent=2),
    )

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["json"]["name"],
        base_url=DEFAULT_CONFIGS["json"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_structured_output(schema)
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    conversation_json = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    # Ensure all strings are unescaped
    if "dialogues" in conversation_json:
        for entry in conversation_json["dialogues"]:
            if "text" in entry:
                entry["text"] = entry["text"].encode("utf-8").decode("unicode-escape")

    return Conversation.model_validate(conversation_json)


@fl.workflow
def podcast_workflow(
    pdf_metadata: list[PDFMetadata],
    speaker_1_name: str,
    speaker_2_name: str,
    retries: int = 5,
    duration: int = 180,
) -> Conversation:
    summarized_pdfs = fl.map_task(
        functools.partial(podcast_summarize_pdf, retries=retries)
    )(pdf_metadata=pdf_metadata)
    raw_outline = podcast_generate_raw_outline(
        summarized_pdfs, duration, retries=retries
    )
    structured_outline = podcast_generate_structured_outline(
        raw_outline, pdf_metadata, retries=retries
    )
    segments = fl.map_task(
        functools.partial(
            podcast_process_segments,
            pdf_metadata={
                "pdf_metadata": pdf_metadata
            },  # list isn't allowed in partial tasks
            retries=retries,
        )
    )(outline_segment=structured_outline.segments)
    dialogues = fl.map_task(
        functools.partial(
            podcast_generate_dialogue,
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
            retries=retries,
        )
    )(
        outline_segment=structured_outline.segments,
        segment=segments,
    )
    dialogue = podcast_combine_dialogues(dialogues, structured_outline, retries=retries)
    conversation = podcast_create_final_conversation(
        dialogue, speaker_1_name, speaker_2_name, retries=retries
    )
    return conversation
