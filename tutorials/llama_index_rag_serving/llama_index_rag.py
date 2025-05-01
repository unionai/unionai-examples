from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from union.app.llm import VLLMApp
from union.app import App
from union import Resources
from flytekit.extras.accelerators import L4

image = "ghcr.io/unionai-oss/serving-vllm:0.1.17"

vllm_embedding_app = VLLMApp(
    name="granite-embedding",
    container_image=image,
    model="flyte://av0.2/demo/thomasjpfan/development/granite-embedding-125m-english@f6d867259e3dc8c27c11f6be5b2b54d0",
    model_id="text-embedding-granite-embedding-125m-english",
    limits=Resources(cpu=7, mem="25Gi", gpu=1),
    requests=Resources(cpu=7, mem="25Gi", gpu=1),
    stream_model=False,
    extra_args="--dtype=half --max-model-len 512",
    scaledown_after=500,
    accelerator=L4,
)

vllm_chat_app = VLLMApp(
    name="qwen-app",
    container_image=image,
    requests=Resources(cpu=7, mem="24Gi", gpu="1"),
    accelerator=L4,
    model="flyte://av0.2/demo/thomasjpfan/development/Qwen2_5-0_5B-Instruct@d7cfd1934f11d4a39b080af0032ec65f",
    model_id="qwen2",
    extra_args="--dtype=half --enable-auto-tool-choice --tool-call-parser hermes",
    scaledown_after=500,
    stream_model=True,
    port=8084,
)


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding = OpenAILikeEmbedding(
        model_name="text-embedding-granite-embedding-125m-english",
        api_key="XYZ",
        api_base=f"{vllm_embedding_app.endpoint}/v1",
        additional_kwargs={"extra_body": {"truncate_prompt_tokens": 512}},
    )

    source_url = "https://en.wikipedia.org/wiki/Star_Wars"

    llm = OpenAILike(
        api_base=f"{vllm_chat_app.endpoint}/v1",
        api_version="v1",
        model="qwen2",
        api_key="XYZ",
        is_chat_model=True,
        is_function_calling_model=True,
        max_tokens=2056,
    )
    Settings.llm = llm
    Settings.embed_model = embedding

    documents = SimpleWebPageReader(html_to_text=True).load_data([source_url])
    vector_store = LanceDBVectorStore(uri="lancedb-serving")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    query_engine = index.as_chat_engine()

    models["query_engine"] = query_engine
    yield


app = FastAPI(lifespan=lifespan)


class MessageInput(BaseModel):
    content: str


@app.post("/ask")
async def ask(message: MessageInput) -> str:
    query_engine = models["query_engine"]
    chat_response = await query_engine.achat(message.content)
    return chat_response.response


from union import ImageSpec

fast_api_image = ImageSpec(
    name="simple-rag-serving",
    requirements="requirements.txt",
    builder="default",
    registry="ghcr.io/unionai-oss",
)

fast_app = App(
    name="llama-index-fastapi",
    container_image=fast_api_image,
    requests=Resources(cpu=2, mem="2Gi"),
    framework_app=app,
    dependencies=[vllm_embedding_app, vllm_chat_app],
    requires_auth=False,
    env={"LANCE_CPU_THREADS": "2"},
)
