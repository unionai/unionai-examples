# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The API client for the langchain-esque service."""

import os
from pathlib import Path
from typing import Generator, Optional, Union

from flytekit.exceptions.base import FlyteRecoverableException
from frontend.utils import (
    EmbeddingConfig,
    LLMConfig,
    RankingConfig,
    RetrieverConfig,
    TextSplitterConfig,
    VectorStoreConfig,
    get_llm,
    get_ranking_model,
    get_vector_store,
)
from requests import ConnectTimeout
from union import UnionRemote

TASK_NAME = "enterprise_rag.ingestion.ingest_docs"
embedding_config = EmbeddingConfig(server_url=os.getenv("EMBEDDING_ENDPOINT"))
llm_config = LLMConfig(
    server_url=os.getenv("LLM_ENDPOINT"), model_name=os.getenv("LLM_MODEL").split(":")[0]
)
reranker_config = RankingConfig(server_url=os.getenv("RERANKER_ENDPOINT"))


def llm_chain(
    query: str,
    chat_history: list[tuple[str, str]],
    llm_config: LLMConfig = llm_config,
) -> Generator[str, None, None]:
    """Using llm to generate response directly without knowledge base."""
    from langchain_core.output_parsers.string import StrOutputParser
    from langchain_core.prompts.chat import ChatPromptTemplate

    system_prompt = """
You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe. 
Please ensure that your responses are positive in nature.
"""

    system_message = [("system", system_prompt)]

    user_message = [("user", "{question}")]
    message = system_message + chat_history + user_message
    prompt_template = ChatPromptTemplate.from_messages(message)

    llm = get_llm(llm_config)

    chain = prompt_template | llm | StrOutputParser()
    return chain.stream({"question": f"{query}"})


def rag_chain_with_multiturn(
    query: str,
    chat_history: list[tuple[str, str]],
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = embedding_config,
    llm_config: LLMConfig = llm_config,
    vector_db_top_k: int = 40,
    top_n: int = 10,
    ranker_config: Optional[RankingConfig] = reranker_config,
    retriever_config: RetrieverConfig = RetrieverConfig(),
    history_count: int = 15,
    enable_querywriter: bool = False,
) -> Generator[str, None, None]:
    from langchain_core.output_parsers.string import StrOutputParser
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.prompts.chat import ChatPromptTemplate
    from langchain_core.runnables import RunnableAssign, RunnablePassthrough

    try:
        vs = get_vector_store(
            os.getenv("MILVUS_URI"),
            os.getenv("MILVUS_TOKEN"),
            vector_store_config,
            embedding_config,
        )
        llm = get_llm(llm_config)
        ranker = get_ranking_model(ranker_config, retriever_config)

        top_k = vector_db_top_k if ranker else top_n
        retriever = vs.as_retriever(
            search_kwargs={"k": top_k}
        )  # milvus does not support similarily threshold

        # conversation is tuple so it should be multiple of two
        # -1 is to keep last k conversation
        history_count = history_count * 2 * -1
        chat_history = chat_history[history_count:]

        system_prompt = ""
        system_prompt = """You are a helpful AI assistant named Envie. 
You will reply to questions only based on the context that you are provided. 
If something is out of context, you will refrain from replying and politely decline to respond to the user.

You are given the following context \n\n
{context}\n\n

Only use the content of the context, do not mention which documents the information is provided from, do not say things like based on provided documents in the final answer and keep the flow conversational.
"""

        system_message = [("system", system_prompt)]

        retriever_query = query
        if enable_querywriter:
            # Based on conversation history recreate query for better document retrieval
            query_rewriter_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
It should strictly be a query not an answer.
"""
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", query_rewriter_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            q_prompt = contextualize_q_prompt | llm | StrOutputParser()
            # query to be used for document retrieval
            retriever_query = q_prompt.invoke(
                {"input": query, "chat_history": chat_history}
            )
            if retriever_query.replace('"', "'") == "''" or len(retriever_query) == 0:
                return iter([""])

        # Prompt for response generation based on context
        user_message = [("user", "{question}")]

        # Prompt template with system message, conversation history and user query
        message = system_message + chat_history + user_message
        prompt = ChatPromptTemplate.from_messages(message)

        if ranker:
            ranker.top_n = top_n
            context_reranker = RunnableAssign(
                {
                    "context": lambda input: ranker.compress_documents(
                        query=input["question"], documents=input["context"]
                    )
                }
            )

            retriever = {
                "context": retriever,
                "question": RunnablePassthrough(),
            } | context_reranker
            docs = retriever.invoke(retriever_query)
            docs = [d.page_content for d in docs.get("context", [])]
            chain = prompt | llm | StrOutputParser()
        else:
            docs = retriever.invoke(retriever_query)
            docs = [d.page_content for d in docs]
            chain = prompt | llm | StrOutputParser()

        return chain.stream({"question": query, "context": docs})
    except ConnectTimeout:
        raise FlyteRecoverableException(
            "Connection timed out while making a request to the LLM endpoint: %s", e
        )


def document_search(
    content: str,
    num_docs: int,
    vector_store_config: VectorStoreConfig = VectorStoreConfig(),
    embedding_config: EmbeddingConfig = embedding_config,
    vector_db_top_k: int = 40,
    ranker_config: Optional[RankingConfig] = reranker_config,
    retriever_config: RetrieverConfig = RetrieverConfig(),
) -> list[dict[str, Union[str, float]]]:
    from langchain_core.runnables import RunnableAssign, RunnablePassthrough

    vs = get_vector_store(
        os.getenv("MILVUS_URI"),
        os.getenv("MILVUS_TOKEN"),
        vector_store_config,
        embedding_config,
    )

    docs = []
    ranker = get_ranking_model(ranker_config, retriever_config)
    top_k = vector_db_top_k if ranker else num_docs
    retriever = vs.as_retriever(
        search_kwargs={"k": top_k}
    )  # milvus does not support similarily threshold

    if ranker:
        # Update number of document to be retriever by ranker
        ranker.top_n = num_docs

        context_reranker = RunnableAssign(
            {
                "context": lambda input: ranker.compress_documents(
                    query=input["question"], documents=input["context"]
                )
            }
        )

        retriever = {
            "context": retriever,
            "question": RunnablePassthrough(),
        } | context_reranker
        docs = retriever.invoke(content)

        resp = []
        for doc in docs.get("context"):
            resp.append(
                {
                    "source": os.path.basename(doc.metadata.get("source", "")),
                    "content": doc.page_content,
                    "score": doc.metadata.get("relevance_score", 0),
                }
            )
        return resp

    docs = retriever.invoke(content)
    resp = []
    for doc in docs:
        resp.append(
            {
                "source": os.path.basename(doc.metadata.get("source", "")),
                "content": doc.page_content,
                "score": doc.metadata.get("relevance_score", 0),
            }
        )
    return resp


class ChatClient:
    """A client for connecting the the lanchain-esque service."""

    def __init__(self, model_name: str) -> None:
        """Initialize the client."""
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the friendly model name."""
        return self._model_name

    def predict(
        self,
        query: str,
        use_knowledge_base: bool,
        chat_history: list[tuple[str, str]],
        num_tokens: int,
    ) -> Generator[str, None, None]:
        chat_history.append({"role": "user", "content": query})
        llm_config.max_tokens = num_tokens

        if use_knowledge_base:
            for chunk in rag_chain_with_multiturn(
                query=query,
                chat_history=chat_history,
                llm_config=llm_config,
            ):
                yield chunk
        else:
            for chunk in llm_chain(
                query=query,
                chat_history=chat_history,
                llm_config=llm_config,
            ):
                yield chunk

    def search(self, prompt: str, num_docs: int) -> list[dict[str, Union[str, float]]]:
        """Search for relevant documents and return json data."""
        return document_search(prompt, num_docs)

    def upload_documents(
        self, file_paths: list[str], version: str
    ) -> tuple[list[str], list[str]]:
        remote = UnionRemote()

        execution_ids = []
        for fpath in file_paths:
            task = remote.fetch_task(name=TASK_NAME)
            execution = remote.execute(
                task,
                inputs={
                    "file_path": remote.upload_file(Path(fpath))[1],
                    "text_splitter_config": TextSplitterConfig(),
                    "vector_store_config": VectorStoreConfig(),
                    "embedding_config": embedding_config,
                },
                version=version,
            )
            execution_ids.append(execution.id.name)
        return (file_paths, execution_ids)

    def delete_documents(
        self,
        file_name: str,
        vector_store_config: VectorStoreConfig = VectorStoreConfig(),
        embedding_config: EmbeddingConfig = embedding_config,
    ) -> bool:
        vector_store = get_vector_store(
            os.getenv("MILVUS_URI"),
            os.getenv("MILVUS_TOKEN"),
            vector_store_config=vector_store_config,
            embedding_config=embedding_config,
        )

        extract_filename = lambda metadata: os.path.basename(metadata["source"])
        milvus_data = vector_store.col.query(
            expr="pk >= 0", output_fields=["pk", "source"]
        )
        ids_list = [
            metadata["pk"]
            for metadata in milvus_data
            if file_name in extract_filename(metadata)
        ]

        if ids_list:
            vector_store.col.delete(f"pk in {ids_list}")
            return file_name
        return False

    def get_uploaded_documents(
        self,
        vector_store_config: VectorStoreConfig = VectorStoreConfig(),
        embedding_config: EmbeddingConfig = embedding_config,
    ) -> list[str]:
        extract_filename = lambda metadata: os.path.basename(metadata["source"])

        vector_store = get_vector_store(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
            vector_store_config=vector_store_config,
            embedding_config=embedding_config,
        )

        # Getting all the ID's > 0
        if vector_store.col:
            milvus_data = vector_store.col.query(
                expr="pk >= 0", output_fields=["pk", "source"]
            )
            filenames = set(extract_filename(metadata) for metadata in milvus_data)
            return list(filenames)
