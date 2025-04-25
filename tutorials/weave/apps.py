import os
import re
import typing
from contextlib import asynccontextmanager

import union
import weave
from fastapi import FastAPI, Request
from flytekit.extras.accelerators import L4
from union.app import App, Input
from union.app.llm import SGLangApp

LLM = "mistralai/Mixtral-8x7B-v0.1"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

llm_image = union.ImageSpec(
    name="mistral-sglang",
    packages=["union-runtime>=0.1.17", "sglang==0.4.5.post3"],
)

sglang_app = SGLangApp(
    name="mistral-sglang",
    container_image=llm_image,
    requests=union.Resources(cpu=12, mem="24Gi", gpu="1"),
    accelerator=L4,
    model_id="mistral-8x7b",
    scaledown_after=300,
    stream_model=True,
)


def extract_filters_from_query(query: str):
    from llama_index.core.vector_stores import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    filters = []

    # Price filters
    if match := re.search(r"(?:under|below)\s*\$?(\d+)", query, re.I):
        filters.append(
            MetadataFilter(
                key="price", value=int(match.group(1)), operator=FilterOperator.LTE
            )
        )

    if match := re.search(r"(?:over|above|more than)\s*\$?(\d+)", query, re.I):
        filters.append(
            MetadataFilter(
                key="price", value=int(match.group(1)), operator=FilterOperator.GTE
            )
        )

    # Number of reviews
    if match := re.search(r"(?:over|more than)\s*(\d+)\s*reviews?", query, re.I):
        filters.append(
            MetadataFilter(
                key="number_of_reviews",
                value=int(match.group(1)),
                operator=FilterOperator.GTE,
            )
        )

    # Superhost
    if re.search(r"\b(superhost|only superhosts)\b", query, re.I):
        filters.append(MetadataFilter(key="host_is_superhost", value="t"))

    # Instant bookable
    if re.search(r"\b(instant book|must be instant bookable)\b", query, re.I):
        filters.append(MetadataFilter(key="instant_bookable", value="t"))

    # Rating filter
    if match := re.search(r"(?:rating|review score)[^\d]*(\d(?:\.\d)?)", query, re.I):
        filters.append(
            MetadataFilter(
                key="review_scores_rating",
                value=float(match.group(1)),
                operator=FilterOperator.GTE,
            )
        )

    # City/Country filter
    if match := re.search(
        r"\b(?:in|around|near|from)\s+([a-zA-Z\s\-']+?)(?:\s+(with|that|which|having|where|for|and|,|\.|$))",
        query,
        re.I,
    ):
        neighborhood = match.group(1).strip()

        # Consider only terms that look like proper nouns (start with uppercase)
        neighborhood_terms = re.split(r"[-,\s]+", neighborhood)

        for term in neighborhood_terms:
            clean_term = term.strip()
            if len(clean_term) > 2 and clean_term[0].isupper():
                filters.append(
                    MetadataFilter(
                        key="neighbourhood",
                        value=clean_term,
                        operator=FilterOperator.TEXT_MATCH_INSENSITIVE,
                    )
                )

    return MetadataFilters(filters=filters)


@asynccontextmanager
async def lifespan(app):
    import weave
    import weaviate
    from llama_index.core import VectorStoreIndex
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from weave.scorers import ContextEntityRecallScorer, ContextRelevancyScorer
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    weave.init()

    llm_client = OpenAI(
        model=LLM,
        base_url=f"{sglang_app.endpoint}/v1",
        api_key="abc",
    )
    app.state.llm_client = llm_client

    vdb_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(timeout=Timeout(init=30)),
    )

    vector_store = WeaviateVectorStore(
        weaviate_client=vdb_client, index_name=os.getenv("INDEX_NAME")
    )
    loaded_index = VectorStoreIndex.from_vector_store(vector_store)
    app.state.query_engine = loaded_index.as_query_engine()

    relevancy_prompt = """
Given the following question and context, rate the relevancy of the context to the question on a scale from 0 to 1.

Question: {question}
Context: {context}
Relevancy Score (0-1):
"""

    app.state.entity_recall_scorer = ContextEntityRecallScorer()
    app.state.relevancy_scorer = ContextRelevancyScorer(
        relevancy_prompt=relevancy_prompt, column_map={"output": "query"}
    )

    yield


def postprocess_inputs(inputs: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {k: v for k, v in inputs.items() if k != "request"}


def postprocess_output(outputs: dict) -> str:
    return outputs["output"]


# Tracing
@weave.op(postprocess_inputs=postprocess_inputs, postprocess_output=postprocess_output)
async def generate_rag_response(query: str, request: Request) -> dict:
    from llama_index.core import Settings
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    Settings.embed_model = FastEmbedEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = request.app.state.llm_client

    query_engine = request.app.state.query_engine
    filters = extract_filters_from_query(query)

    response = query_engine.query(query, filters=filters)

    context = []
    for node in response.source_nodes:
        context.append(node.get_content())

    return {
        "output": response.response,
        "context": context,
    }


app_image = union.ImageSpec(
    name="weave-rag",
    packages=[
        "openai==1.74.0",
        "llama-index==0.12.32",
        "llama-index-vector-stores-weaviate==1.3.1",
        "llama-index-embeddings-fastembed==0.3.1",
        "fastapi[standard]==0.115.12",
        "weave[scorers]==0.51.43",
        "union-runtime>=0.1.17",
    ],
)

fastapi_app = FastAPI(lifespan=lifespan)
app = App(
    name="weave-rag",
    secrets=[
        union.Secret(key="wandb_api_key", env_var="WANDB_API_KEY"),
        union.Secret(key="weaviate_url", env_var="WEAVIATE_URL"),
        union.Secret(key="weaviate_api_key", env_var="WEAVIATE_API_KEY"),
    ],
    env={"WEAVE_USE_SERVER_CACHE": True, "WANDB_PROJECT": "mistral-sglang"},
    dependencies=[sglang_app],
    framework_app=fastapi_app,
    container_image=app_image,
    inputs=[Input(name="index_name", type=str, env_var="INDEX_NAME")],
    scaledown_after=300,
)


@fastapi_app.post("/query_rag")
async def query_rag(query: str, request: Request) -> str:
    result, call = await generate_rag_response.call(query, request)

    # Guardrails
    recall_score = await call.apply_scorer(
        request.app.state.entity_recall_scorer, {"context": result["context"]}
    )
    recall_threshold = 0.7
    if recall_score.result["recall"] < recall_threshold:
        return "I'm unable to respond confidently because the answer isn't well-supported by the retrieved context."

    relevancy_score = await call.apply_scorer(
        request.app.state.relevancy_scorer, {"context": result["context"]}
    )
    relevancy_threshold = 0.4
    if relevancy_score.result.relevancy_score < relevancy_threshold:
        return f"The context is not relevant to the question. Reasoning: {relevancy_score.result.reasoning}"

    return result["output"]
