import functools
import json
from typing import Literal, Optional

import flytekit as fl
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel
from union.actor import ActorEnvironment

from .monologue_prompts import FinancialSummaryPrompts
from .utils import image, nvidia_key

# TODO: Unable to host models due to the need for larger GPUs (serverless is not a viable option then)
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

monologue_actor = ActorEnvironment(
    name="monologue-flow-actor",
    replica_count=1,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=nvidia_key)],
    container_image=image,
    requests=fl.Resources(mem="10Gi", cpu="5"),
)


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


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_summarize_pdfs(pdf_metadata: PDFMetadata, retries: int) -> PDFMetadata:
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


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_generate_raw_outline(
    summarized_pdfs: list[PDFMetadata],
    guide_prompt: str,
    retries: int = 5,
) -> str:
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


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_generate_monologue(
    pdf_metadata: list[PDFMetadata],
    raw_outline: str,
    speaker_1_name: str,
    guide_prompt: str,
    retries: int = 5,
) -> str:
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


@monologue_actor.task(cache=True, cache_version="0.1")
def monologue_create_final_conversation(
    monologue: str,
    speaker_1_name: str,
    retries: int = 5,
) -> Conversation:
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
