# # From NVIDIA Blueprint to Union Workflow: Building a PDF-to-Podcast Pipeline
#
# ## Overview
# This documentation demonstrates how to leverage Union to productionize NVIDIA blueprint workflows efficiently and at scale.
# The example focuses on a PDF to podcast conversion workflow, showcasing how Union simplifies the deployment and management of NVIDIA blueprint agents.
#
# ## What are NVIDIA blueprint agents?
# NVIDIA blueprint agents are sophisticated orchestration tools that combine microservices and AI agents to deliver specialized AI workflows. They provide:
#
# - Pre-configured AI service templates
# - Integrated AI model deployment
# - Service orchestration capabilities
#
# ## The Union advantage
# While NVIDIA Launchables offer quick Blueprint deployments, enterprise-scale operations require more robust solutions.
# Union addresses key operational challenges and extends Blueprint capabilities for production environments.
#
# ## Common blueprint implementation challenges
# 1. **Architectural complexity**
# Traditional blueprint implementations often involve:
#
# - Complex service interactions requiring deep architectural understanding
# - Scattered boilerplate code across multiple services
# - Time-intensive customization processes
# - Limited visibility into cross-service dependencies
#
# 2. **Operational hurdles**
# Production deployments face several challenges:
#
# - Resource-intensive scaling requirements
# - Limited built-in error handling capabilities
# - Complex multi-user management
# - Infrastructure overhead for enterprise deployment
#
# ## How Union enhances blueprint workflows
# 1. **Simplified architecture**
#
# - Unified workflow view
# - Streamlined service integration
# - Reduced boilerplate code
# - Clear service dependency mapping
#
# 2. **Production-ready features**
#
# - Built-in scaling capabilities
# - Advanced error handling
# - Multi-tenant support
# - Enterprise-grade orchestration
#
# ## Implementation guide
# This example takes NVIDIA's [PDF-to-podcast blueprint agent](https://github.com/NVIDIA-AI-Blueprints/pdf-to-podcast) and adapts it into a Union workflow.
#
# First, let's import all the necessary dependencies.

import functools
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Literal, Optional

import flytekit as fl
import pandas as pd
from flytekit.types.file import FlyteFile
from pydantic import BaseModel
from union import ActorEnvironment
from union.actor import ActorEnvironment

from .monologue_prompts import FinancialSummaryPrompts
from .podcast_prompts import PodcastPrompts
from .utils import eleven_key, image, nvidia_key

# We use different language models for different tasks. Here's our default configuration:

DEFAULT_CONFIGS = {
    "reasoning": {
        "name": "meta/llama-3.1-405b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
    },
    "iteration": {
        "name": "meta/llama-3.1-405b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
    },
    "json": {
        "name": "meta/llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
    },
}

# Let's break down what each model will help us accomplish:
# - For complex reasoning tasks, we use the 405B parameter Llama model
# - For iterative refinement, we also use the 405B parameter model
# - For structured JSON output, we use the 70B parameter model
#
# # Monologue workflow
#
# Let's create a workflow that transforms PDF documents into a polished monologue script.
# - We begin by using the `monologue_summarize_pdfs` task to process your input PDFs.
# - Next, we use `monologue_generate_raw_outline` to structure the content.
# - With our outline ready, `monologue_generate_monologue` creates the initial script.
# - Finally, `monologue_create_final_conversation` polishes the script.
#
# The workflow runs in an [actor](https://docs.union.ai/byoc/user-guide/core-concepts/actors) environment configured for these tasks.

monologue_actor = ActorEnvironment(
    name="monologue-flow-actor",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=nvidia_key)],
    container_image=image,
    requests=fl.Resources(mem="10Gi", cpu="5"),
)

# This configuration:
#
# - Allocates 10GB of memory and 5 CPU cores
# - Sets a 5-minute (300 seconds) time-to-live
# - Uses the NVIDIA API key for model access
# - Runs a single replica of the workflow
#
# We define these Pydantic models to structure our data:


class PDFMetadata(BaseModel):
    filename: str
    status: str
    type: str
    content: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[str] = None


class DialogueEntry(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2"]


class Conversation(BaseModel):
    scratchpad: str
    dialogue: list[DialogueEntry]


# Each model serves a specific purpose:
#
# - `PDFMetadata`: Tracks document processing status and content
# - `DialogueEntry`: Structures individual dialogue segments
# - `Conversation`: Manages the overall conversation flow
#
# We begin by summarizing the PDFs.


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_summarize_pdfs(pdf_metadata: PDFMetadata, retries: int) -> PDFMetadata:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    template = FinancialSummaryPrompts.get_template("monologue_summary_prompt")

    prompt = template.render(text=pdf_metadata.content)

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["reasoning"]["name"],
        base_url=DEFAULT_CONFIGS["reasoning"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    resp = llm.invoke(
        [{"role": "user", "content": prompt}],
    )
    pdf_metadata.summary = resp.content
    return pdf_metadata


# The task processes the provided PDF metadata, which includes the document's content, and generates a concise summary based on the information in the PDF.
# Using the Llama 405B model, we configure the task with the necessary API keys to access NVIDIA's endpoints.
# The generated summary is then stored back into the `pdf_metadata` object, updating it with the newly created content.
#
# Once we have the summaries, the next step is to create a raw outline based on the summarized content.


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_generate_raw_outline(
    summarized_pdfs: list[PDFMetadata],
    guide_prompt: str,
    retries: int = 5,
) -> str:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    documents = [f"Document: {pdf.filename}\n{pdf.summary}" for pdf in summarized_pdfs]

    template = FinancialSummaryPrompts.get_template(
        "monologue_multi_doc_synthesis_prompt"
    )
    prompt = template.render(
        focus_instructions=guide_prompt, documents="\n\n".join(documents)
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


# The task takes in a list of summarized PDFs and a guide prompt. Using the Llama 405B model,
# we synthesize the summarized information into a structured outline. The prompt, sourced from the `FinancialSummaryPrompts` class,
# helps organize the content in a way that provides a clear framework for what comes next.
#
# After generating the raw outline, we move to creating a full monologue script.


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_generate_monologue(
    pdf_metadata: list[PDFMetadata],
    raw_outline: str,
    speaker_1_name: str,
    guide_prompt: str,
    retries: int = 5,
) -> str:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
    prompt = template.render(
        raw_outline=raw_outline,
        documents=pdf_metadata,
        focus=guide_prompt,
        speaker_1_name=speaker_1_name,
    )

    llm = ChatNVIDIA(
        model=DEFAULT_CONFIGS["reasoning"]["name"],
        base_url=DEFAULT_CONFIGS["reasoning"]["api_base"],
        nvidia_api_key=fl.current_context().secrets.get(key=nvidia_key),
        max_tokens=None,
    )
    llm = llm.with_retry(stop_after_attempt=retries, wait_exponential_jitter=True)
    monologue = llm.invoke(
        [{"role": "user", "content": prompt}],
    )

    return monologue.content


# This task is responsible for turning the raw outline into a full monologue script.
# It uses the outline, the summarized PDFs, the name of the speaker, and a guide prompt to generate a cohesive narrative.
# Again, we rely on the Llama 405B model, which we configure to generate a script that flows naturally and communicates the extracted
# financial insights clearly. We use a template from the `FinancialSummaryPrompts` class to ensure the content is engaging and easy to follow.
#
# Once the monologue script is ready, we convert it into a conversation.


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_create_final_conversation(
    monologue: str,
    speaker_1_name: str,
    retries: int = 5,
) -> Conversation:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    schema = Conversation.model_json_schema()
    template = FinancialSummaryPrompts.get_template("monologue_dialogue_prompt")
    prompt = template.render(
        speaker_1_name=speaker_1_name,
        text=monologue,
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


# In the `monologue_create_final_conversation` task, we take the monologue script and the speaker's name,
# and structure them as a conversation. Using a dialogue template from the `FinancialSummaryPrompts` class, we ensure the monologue
# is formatted into a coherent conversation. We leverage the Llama 405B model’s structured output capabilities to create the conversation schema,
# and we handle any necessary string unescaping to ensure the formatting is correct.
#
# We bring everything together in the `monologue_workflow` workflow.


@fl.workflow
def monologue_workflow(
    pdfs: list[PDFMetadata],
    speaker_1_name: str,
    guide_prompt: str = "key financial metrics and performance indicators",
    retries: int = 5,
) -> Conversation:
    summarized_pdfs = fl.map_task(
        functools.partial(monologue_summarize_pdfs, retries=retries)
    )(pdf_metadata=pdfs)
    raw_outline = monologue_generate_raw_outline(
        summarized_pdfs, guide_prompt=guide_prompt, retries=retries
    )
    monologue = monologue_generate_monologue(
        pdf_metadata=summarized_pdfs,
        raw_outline=raw_outline,
        speaker_1_name=speaker_1_name,
        guide_prompt=guide_prompt,
        retries=retries,
    )
    conversation = monologue_create_final_conversation(
        monologue=monologue, speaker_1_name=speaker_1_name, retries=retries
    )
    return conversation


# This workflow takes in a list of `PDFMetadata` objects (representing the PDFs to be processed), a speaker’s name, and an optional guide prompt
# (with a default prompt focusing on key financial metrics).
# It returns the conversation object, which can then be used for further processing or as output.
#
# ## Podcast workflow
#
# The `podcast_workflow` orchestrates the creation of a podcast by taking in a list of `PDFMetadata` objects, speaker names, and a duration.
# The workflow returns a conversation object that forms the backbone of the podcast.
# To achieve this, we use a series of tasks that work together to generate the podcast outline and its segments, which are then used to create the final conversation.
#
# We define the actor environment where the tasks will run. The actor enables us to reuse the same container across multiple tasks.

podcast_actor = ActorEnvironment(
    name="podcast-flow-actor",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=nvidia_key)],
    container_image=image,
    requests=fl.Resources(mem="10Gi", cpu="5"),
)

# Next, we define some Pydantic models to structure our data:


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


# The segment models define the structure of the podcast outline. These models help organize the various topics and segments.
#
# We begin by summarizing each PDF using the `podcast_summarize_pdf` task.


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


# This task processes the content of each PDF and extracts key insights, providing concise summaries.
# These summaries lay the groundwork for the rest of the podcast creation process.
#
# Once we have the summaries, we generate the raw outline using the `podcast_generate_raw_outline` task.


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


# This step synthesizes the summarized content into a rough structure, highlighting the key themes and points for discussion.
#
# Next, we refine the raw outline using the `podcast_generate_structured_outline` task.


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


# It takes the raw outline and organizes it into a more formal, structured format.
# This ensures the podcast flows logically, making it easier for the audience to follow and engage with the content.
#
# We then break down the structured outline into individual segments using the `podcast_process_segments` task.


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


# This task processes each segment, referencing content from the PDFs and enriching the segment with relevant information,
# ensuring the podcast is informative and engaging.
#
# After processing the segments, we generate the dialogue using the `podcast_generate_dialogue` task.


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


# This task formats the segment content into a conversation between two speakers, ensuring that the dialogue flows naturally, remains engaging,
# and stays true to the outlined content.
#
# The `podcast_combine_dialogues` task stitches the dialogues from different segments into a cohesive conversation.


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


# It ensures smooth transitions between topics and guarantees that the overall dialogue feels like a continuous, engaging narrative.
#
# Finally, we define the `podcast_create_final_conversation` task to format the combined dialogue into a structured conversation model.


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


# This task takes the combined dialogue and formats it according to a structured conversation model,
# ensuring that the final product is polished, professional, and ready for presentation.
#
# Finally, we define the `podcast_workflow` workflow, which orchestrates the entire podcast workflow.


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


# This workflow defines the steps for creating a podcast from a list of PDF documents.
# It processes the PDFs, generates outlines, creates dialogues, and assembles the final podcast conversation.
#
# ## PDF conversion and text-to-speech workflow
# The PDF class represents a PDF document and its associated metadata.
# We define two attributes in the class: `type` and `pdf`. The type attribute specifies the category of the PDF, such as "target" or "context."
# The pdf attribute contains the `FlyteFile`, which points to the actual PDF document we will process in the workflow.


class PDF(BaseModel):
    type: str
    pdf: FlyteFile


# The `tts_actor` environment defines the resources and settings for the Text-to-Speech (TTS) tasks.
# We configure the actor with one replica and set a time-to-live (TTL) of 300 seconds to ensure timely execution.

tts_actor = ActorEnvironment(
    name="tts",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=eleven_key)],
    container_image=image.with_packages(
        ["elevenlabs", "flytekitplugins-deck-standard"]
    ),
)

# The actor accesses the Eleven Labs API for voice synthesis, and it requires an API key stored as a secret.
#
# The `convert_pdf` task processes a given PDF document and converts it into structured metadata.
# We use the `DocumentConverter` to extract content from the PDF, handling any conversion errors along the way. This task performs the following operations:
# 1. Downloads the provided PDF file and passes it to the `DocumentConverter` to perform the conversion.
# 2. Based on the conversion result, it assigns a status of "success" or "failed." If the conversion is successful,
#    we extract the document’s content and store it; otherwise, we capture any errors that occurred during the conversion.
# 3. Finally, the task returns a `PDFMetadata` object containing the filename, status, extracted content (if successful),
#    and any error messages (if conversion fails).


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


# We use this task as the first step in transforming PDF files into usable content for the podcast.
#
# Next, we define a task that retrieves a list of available voices from the Eleven Labs API.
# We use this task to fetch detailed information about each voice, such as its name, category, accent, gender, and use case. We perform the following actions:
# 1. Initialize the Eleven Labs client using the API key provided in the secrets.
# 2. Make a request to the Eleven Labs API to fetch all available voices.
# 3. Organize the voice data into a structured format, including voice properties such as the voice's name, accent, and description.
# 4. Display the collected voice data as an HTML table using Flyte deck's `TableRenderer` to make it easy to review and select voices for the podcast.


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


# This task helps us explore the available voices before we choose which ones to use for the dialogue synthesis.
#
# We define a `convert_text_to_voice` task to generate audio from a given text dialogue entry.
# This task converts dialogue text into speech using Eleven Labs’ Text-to-Speech API. We carry out the following steps:
# 1. Initialize the Eleven Labs client with the required API key.
# 2. Determine which voice to use based on the speaker’s identity. We use the `speaker_names` dictionary to select the correct voice for either speaker-1 or speaker-2.
# 3. Use the Eleven Labs API to generate an audio stream from the provided text. We pass the selected voice, model, and voice settings to the API to
#    control factors like stability and similarity boost.
# 4. The task returns the generated audio as a byte stream.


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


# We then define a `coalesce_dialogues` task that takes multiple audio streams, combines them, and produces a single audio file. We do this by:
#
# 1. Creating a podcast.mp3 file in the working directory.
# 2. Writing each byte stream from the provided list of dialogues into the file.
# 3. Once all audio chunks are written, we return a `FlyteFile` object pointing to the final audio file.


@tts_actor.task
def coalesce_dialogues(dialogues: list[bytes]) -> FlyteFile:
    working_dir = Path(fl.current_context().working_directory)
    podcast_file_path = working_dir / "podcast.mp3"

    with open(podcast_file_path, "wb") as file:
        for chunk in dialogues:
            file.write(chunk)

    return FlyteFile(path=podcast_file_path)


# This task allows us to merge all generated audio segments into a complete podcast file, ready for distribution.
#
# Finally, we define the `pdf_to_podcast` workflow.


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


# We use this workflow to process PDFs, select speaker voices, generate dialogues, and synthesize the final podcast audio.
#
# ## Key benefits of using Union
# With Union, we’re able to simplify and optimize each of these steps, making the entire process more efficient and easier to manage.
#
# 1. **Microservices as tasks**
#    Union enables translating blueprint microservices into Flyte tasks. Instead of setting up API services,
#    tasks can run in Kubernetes pods (or shared pods with Union actors), with independent scaling and resource configurations.
#    No APIs—just clean Python functions. Tasks can also be executed independently, just like the microservices they replace.
#    For example, in this workflow, the monologue and dialogue are separate workflows that can be run independently.
#    With Union, you can modularize your code while maintaining a single-pane view of the entire workflow, providing both flexibility and clarity.
# 2. **Built-in infrastructure**
#    Union comes with out-of-the-box solutions for background jobs, queuing, and storage (e.g., Redis, Celery, MinIO).
#    It supports multi-tenancy, concurrent workflows, and automatic data storage in blob storage, so you don’t need to manage these yourself.
# 3. **Monitoring and logging**
#    - Automatic workflow versioning.
#    - Built-in **data lineage** tracks the flow of data within workflows, forming an organizational graph for observability.
#    - Real-time logging with integration options for custom loggers, eliminating the need for custom trackers.
#    - Because microservices are translated into Flyte tasks, you can avoid the complexity of tools like Jaeger.
# 4. **Scalability**
#    Union is designed for production-grade performance and scalability. It supports parallel workflows, tracks data, and manages triggers,
#    making it an ideal choice for high-throughput tasks like podcast generation. You can independently scale each task,
#    simplifying the architecture while maintaining performance and flexibility.
# 5. **Parallelism**
#    Union handles up to 100,000 nested parallel tasks. For example, tasks like summarizing PDFs, processing segments, and generating markdown can run
#    simultaneously, leading to significant speed improvements.
# 6. **Secrets management**
#    You can define and manage secrets directly within the SDK—no need for external secrets managers.
# 7. **Human-in-the-loop**
#    Workflows can include human inputs. For example, in the PDF-to-podcast workflow, users can select speakers from a visual deck without needing to
#    manually provide IDs.
# 8. **Error handling and retries**
#    Union’s built-in retries handle transient errors, such as external service downtimes, automatically
#    (given we raise flyte-specific retry error). This eliminates boilerplate code and simplifies error management.
# 9. **Simplified Image Management**
#    Use `ImageSpec` to define images directly in Python—no need to mess with Dockerfiles. Customization becomes quick and easy.
# 10. **Caching**
#     Cache outputs of tasks to reuse results for identical inputs, dramatically improving execution speed and efficiency.
# 11. **Secure**
#     Union is SOC II compliant, ensuring robust data protection and compliance with security standards.
#
# Union simplifies the **productionizing of NVIDIA Blueprint workflows** by making them scalable and efficient.
# Developers can focus on building business logic, while Union handles compute and infrastructure.
# Plus, you can host NIMs locally in Union with the [NIM integration](https://www.union.ai/blog-post/union-powers-faster-end-to-end-ai-application-deployment-using-nvidia-nim).
