from dataclasses import dataclass
from typing import Literal, Optional

from langchain.llms.base import LLM
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.vectorstores import VectorStore
from mashumaro.mixins.json import DataClassJSONMixin


@dataclass
class LLMConfig(DataClassJSONMixin):
    server_url: str = ""
    model_name: str = "meta/llama3-70b-instruct"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class RankingConfig(DataClassJSONMixin):
    model_name: str = "nv-rerank-qa-mistral-4b:1"
    model_engine: str = "nvidia-ai-endpoints"
    server_url: str = ""


@dataclass
class RetrieverConfig(DataClassJSONMixin):
    top_k: int = 4
    score_threshold: float = 0.25
    nr_pipeline: Literal["ranked_hybrid", "hybrid"] = "ranked_hybrid"

    def __post_init__(self):
        allowed_pipelines = {"ranked_hybrid", "hybrid"}
        if self.nr_pipeline not in allowed_pipelines:
            raise ValueError(
                f"nr_pipeline must be one of {allowed_pipelines}, got '{self.nr_pipeline}'"
            )


@dataclass
class VectorStoreConfig(DataClassJSONMixin):
    nlist: int = 64
    nprobe: int = 16
    index_type: str = "FLAT"  # Used to be GPU_IVF_FLAT
    collection_name: str = "enterprise_rag_vectordb"


@dataclass
class TextSplitterConfig(DataClassJSONMixin):
    model_name: str = "Snowflake/snowflake-arctic-embed-l"
    chunk_size: int = 510
    chunk_overlap: int = 200


@dataclass
class EmbeddingConfig(DataClassJSONMixin):
    model_name: str = "snowflake/arctic-embed-l"
    model_engine: str = "nvidia-ai-endpoints"
    dimensions: int = 1024
    server_url: str = ""

    def __post_init__(self):
        if self.model_engine not in {"nvidia-ai-endpoints", "huggingface"}:
            raise ValueError(
                f"Invalid model_engine: {self.model_engine}. Allowed values: 'nvidia-ai-endpoints', 'huggingface'."
            )


def get_embedding_model(embedding_config: EmbeddingConfig) -> Embeddings:
    import torch

    model_kwargs = {"device": "cpu"}
    if torch.cuda.is_available():
        model_kwargs["device"] = "cuda:0"

    encode_kwargs = {"normalize_embeddings": False}

    if embedding_config.model_engine == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        hf_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return hf_embeddings

    if embedding_config.model_engine == "nvidia-ai-endpoints":
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        if embedding_config.server_url:
            return NVIDIAEmbeddings(
                base_url=f"{embedding_config.server_url}/v1",
                model=embedding_config.model_name,
                truncate="END",
            )
        return NVIDIAEmbeddings(model=embedding_config.model_name, truncate="END")


def get_vector_store(
    uri: str,
    token: str,
    vector_store_config: VectorStoreConfig,
    embedding_config: EmbeddingConfig,
) -> VectorStore:
    from langchain_milvus import Milvus

    vector_store = Milvus(
        embedding_function=get_embedding_model(embedding_config),
        connection_args={
            "uri": uri,
            "token": token,
            "secure": True,
        },
        collection_name=vector_store_config.collection_name,
        index_params={
            "index_type": vector_store_config.index_type,
            "metric_type": "L2",
            "nlist": vector_store_config.nlist,
        },
        search_params={"nprobe": vector_store_config.nprobe},
        auto_id=True,
    )
    return vector_store


def get_llm(llm_config: LLMConfig) -> LLM | SimpleChatModel:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    if llm_config.server_url:
        return ChatNVIDIA(
            base_url=f"{llm_config.server_url}/v1",
            model=llm_config.model_name,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            max_tokens=llm_config.max_tokens,
        )

    return ChatNVIDIA(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        max_tokens=llm_config.max_tokens,
    )


def get_ranking_model(
    ranking_config: RankingConfig, retriever_config: RetrieverConfig
) -> BaseDocumentCompressor:
    from langchain_nvidia_ai_endpoints import NVIDIARerank

    if ranking_config.server_url:
        return NVIDIARerank(
            base_url=f"{ranking_config.server_url}/v1",
            top_n=retriever_config.top_k,
            truncate="END",
        )

    if ranking_config.model_name:
        return NVIDIARerank(
            model=ranking_config.model_name,
            top_n=retriever_config.top_k,
            truncate="END",
        )
