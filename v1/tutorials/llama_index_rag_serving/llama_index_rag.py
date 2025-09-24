# # Serving LlamaIndex RAG with FastAPI and LanceDB
#
# This tutorial shows you how to serve a LLamaIndex RAG with FastAPI, where the data
# source is the Wikipedia page for Star Wars.

# {{run-on-union}}

# With Union's App Serving SDK, we define three applications:
# - VLLM backed app hosting the granite-embedding to embedded our data.
# - VLLM backed app hosting Quen2.5 LLM to process the context
# - FastAPI app that has one `/ask` endpoint that uses the above two apps to answer questions about Star Wars
#
# We start by importing the necessary libraries and modules for this example:

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
from union import Resources, ImageSpec
from flytekit.extras.accelerators import L4
import os

# ## Caching From HuggingFace
#
# First, we cache the model weights from HuggingFace, which requires a hugging face secret api key:
#
# 1. **Generate an API key:** Obtain your API key from the Hugging Face website.
# 2. **Create a Secret:** Use the Union CLI to create the secret:
#
# ```shell
# $ union create secret hf-api-key
# ```
#
# After creating your HuggingFace API key, we cache the two models by running:
#
# ```shell
# $ union cache model-from-hf ibm-granite/granite-embedding-125m-english --hf-token-key  hf-api-key \
# $   --union-api-key EAGER_API_KEY --cpu 2 --mem 6Gi --wait
# $ union cache model-from-hf Qwen/Qwen2.5-0.5B-Instruct --hf-token-key  hf-api-key \
# $   --union-api-key EAGER_API_KEY --cpu 2 --mem 6Gi --wait
# ```
# In both cases, you'll get an artifact ID, please copy and replace the artifact URIs here:
EMBEDDING_ARTIFACT = "flyte://av0.2/demo/thomasjpfan/development/granite-embedding-125m-english-20@2abf07bdc0d2bcd17715afeeb293ebfb"
LLM_ARTIFACT = "flyte://av0.2/demo/thomasjpfan/development/Qwen2_5-0_5B-Instruct-20@0ed210eb88c94c16f17d4697a0a91212"

# ## Configuring the VLLM Apps
#
# We configure the two VLLM Apps to serve both the embedding model and the LLM with `VLLMApp` and defining it's resources
# such as the CPU, Memory and accelerator type. With `stream_model=True` the model is streamed from the object store
# to the GPU.
image = "ghcr.io/unionai-oss/serving-vllm:0.1.17"

vllm_embedding_app = VLLMApp(
    name="granite-embedding",
    container_image=image,
    model=EMBEDDING_ARTIFACT,
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
    model=LLM_ARTIFACT,
    model_id="qwen2",
    extra_args="--dtype=half --enable-auto-tool-choice --tool-call-parser hermes",
    scaledown_after=500,
    stream_model=True,
    port=8084,
)

# ## Defining the FastAPI App
#
# The FastAPI App uses a `lifespan` to define the startup behavior. Here we create a LlamaIndex query engine
# using the embedder to convert wikiedpia artifact into vectors and placing it into a LanceDB database.

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


# Here we define a simple `/ask` API to query the RAG and returning the response.
class MessageInput(BaseModel):
    content: str


@app.post("/ask")
async def ask(message: MessageInput) -> str:
    query_engine = models["query_engine"]
    chat_response = await query_engine.achat(message.content)
    return chat_response.response


# ## Configuring the Image for FastAPI
#
# Next, we use an ImageSpec to build an image with dependencies defined in `requirements.txt`:
# In your environment set, `IMAGE_SPEC_REGISTRY` to your registry:`

fast_api_image = ImageSpec(
    name="simple-rag-serving",
    requirements="requirements.txt",
    builder="default",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
)

# We declare that the FastAPI app depends on both the embedding app and the LLM app by setting
# the `dependencies` parameter. The `framework_app` configures the `App` object to load the `FastAPI`
# app as the entrypoint point:

fast_app = App(
    name="llama-index-fastapi",
    container_image=fast_api_image,
    requests=Resources(cpu=2, mem="2Gi"),
    framework_app=app,
    dependencies=[vllm_embedding_app, vllm_chat_app],
    requires_auth=False,
    env={"LANCE_CPU_THREADS": "2"},
    scaledown_after=500,
)

# Deploy the app by running:
#
# ```shell
# $ union deploy apps llama_index_rag.py llama-index-fastapi
# ...
# 🚀 Deployed Endpoint: https://<union-url>
# ````
#
# To query the endpoint with curl run:
#
# ```shell
# $ curl -X 'POST' \
# $  'https://<union-url>/ask' \
# $  -H 'accept: application/json' \
# $  -H 'Content-Type: application/json' \
# $  -d '{
# $     "content": "What was the title of the first Star Wars Movie?"
# $   }'
# ```
#
# Which return an answer to the query:
#
# ```shell
# "The first Star Wars movie was titled \"A New Hope.\""
# ```
