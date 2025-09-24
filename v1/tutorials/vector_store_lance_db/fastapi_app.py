import os
from contextlib import asynccontextmanager
from typing import TypedDict

import lancedb
import pyarrow
from google import genai
from google.genai import types
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.2


class RAGComponents(TypedDict):
    embedding_model: SentenceTransformer
    generation_client: genai.Client
    papers_table: lancedb.table.AsyncTable
    paper_ids_table: lancedb.table.AsyncTable


components = RAGComponents()


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_store_path = os.environ["VECTOR_STORE_PATH"]
    db = await lancedb.connect_async(vector_store_path)
    components["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL)
    components["generation_client"] = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    components["papers_table"] = await db.open_table("papers")
    components["paper_ids_table"] = await db.open_table("paper_ids")
    yield


app = FastAPI(lifespan=lifespan)


async def retrieve(query: str, paper_id: str | None = None, top_k: int = 10):
    """Retrieve the top k papers from the vector store."""
    model = components["embedding_model"]
    papers_table = components["papers_table"]

    vector_query = model.encode(query)
    query = await papers_table.search(vector_query, "vector")
    if paper_id is not None:
        query = query.where(f"paper_id = '{paper_id}'")
    result = await query.limit(top_k).to_list()
    return result


async def generate(query: str, context: list[str]):
    """Generate a streaming response to the query using the context."""

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    client = components["generation_client"]
    async for chunk in await client.aio.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        ),
    ):
        delta = chunk.candidates[0].content.parts[0].text
        if delta is None:
            yield ""
            break
        yield delta


@app.get("/")
async def root():
    """Return a message to the user."""
    return {"message": "Arxiv RAG API"}


@app.get("/papers")
async def papers() -> list[dict]:
    papers_ids_table = components["paper_ids_table"]
    result: pyarrow.Table = await papers_ids_table.to_arrow()
    return result.to_pylist()


@app.get("/ask_paper/{paper_id}")
async def ask(paper_id: str, query: str, top_k: int = 10) -> StreamingResponse:
    context = await retrieve(query, paper_id, top_k=top_k)
    return StreamingResponse(
        generate(query, context),
        media_type="text/event-stream",
    )


@app.get("/ask")
async def ask(query: str, top_k: int = 10) -> StreamingResponse:
    context = await retrieve(query, top_k=top_k)
    return StreamingResponse(
        generate(query, context),
        media_type="text/event-stream",
    )
