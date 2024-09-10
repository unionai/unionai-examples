# # Agentic Retrieval Augmented Generation
#
# This tutorial demonstrates how to implement an agentic retrieval-augmented
# generation (RAG) workflow on Union using [Langchain](https://python.langchain.com/v0.2/docs/introduction/)
# for the RAG building blocks, [ChromaDB](https://www.trychroma.com/) as the
# vector store, and [GPT4](https://openai.com/) for the language model.

# ## Overview
#
# What makes a RAG workflow agentic? The main idea is that the workflow needs to
# define some kind of LLM-mediated process that can make decisions about the
# runtime structure of the workflow. For example, an agent could take a user's
# prompt and decide that it needs to use tools like a calculator, retrieve
# documents from a vector store, or rewrite the user's original query.
#
# In this tutorials we'll build a biomedical research assistant that is able to:
#
# - Use a retrieval tool to fetch documents from [PubMed](https://pubmed.ncbi.nlm.nih.gov/),
#   a repository of biomedical research.
# - Grade the relevance of those documents against the user's query
# - Potentially rewrite the user's original query
# - Generate a final answer.
#
# First, let's import the workflow dependencies:

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Annotated, Optional

from flytekit import dynamic, task, workflow, Artifact, Secret
from flytekit import ImageSpec
from flytekit.types.directory import FlyteDirectory
from union.actor import ActorEnvironment
from utils import env_secret, use_pysqlite3


openai_env_secret = partial(
    env_secret,
    secret_name="openai_api_key",
    env_var="OPENAI_API_KEY",
)

# maximum number of question rewrites
MAX_REWRITES = 10

# ## Creating Secrets for an OpenAI API key
#
# Go to the [OpenAI website](https://platform.openai.com/api-keys) to get an
# API key. Then, create a secret with the `union` CLI tool:
#
# ```bash
# union create secret openai_api_key
# ```
#
# then paste the client ID when prompted. We'll use the `openai_api_key` secret
# throughout this tutorial to authenticate with the OpenAI API and use GPT4 as
# the underlying LLM.

# ## Defining the container image
#
# Here we define the container image that the RAG workflow will run on, pinning
# dependencies to ensure reproducibility.

image = ImageSpec(
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    packages=[
        "beautifulsoup4==4.12.3",
        "chromadb==0.5.3",
        "langchain==0.2.11",
        "langchain-community==0.2.6",
        "langchain-openai==0.1.14",
        "langchain-text-splitters==0.2.2",
        "langchainhub==0.1.20",
        "pysqlite3-binary",
        "tiktoken==0.7.0",
        "xmltodict==0.13.0",
    ],
)

# ## Creating an `ActorEnvironment`
#
# In order to run our RAG workflow quickly, we define an `ActorEnvironment` so
# that we can reuse the container to run the steps of our workflow. We can specify
# variables like:
#
# - `replica_count`: how many workers to provision to run tasks.
# - `parallelism`: the number of tasks that can run in parallel per worker.
# - `ttl_seconds`: how long to keep the actor alive while no tasks are being run.

actor = ActorEnvironment(
    name="agentic-rag",
    replica_count=1,
    ttl_seconds=60,
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)

# ## Creating a vector store `Artifact`
#
# We also use Union `Artifact`s to persist the vector store of documents. We do
# this by defining a `AgenticRagVectorStore` artifact, which is used in our RAG
# workflow to retrieve documents to help answer the user's query.
#
# The `create_vector_store` task below creates the vector store artifact,
# which uses Langchain's `PubMedLoader` to fetch `n` number of documents,
# defined by the `load_max_docs` parameter, based on the `query` provided.
#
# We then split the documents with the `RecursiveCharacterTextSplitter` and
# persist it in a `Chroma` vector store database, which we store as a
# `FlyteDirectory` and annotate with `AgenticRagVectorStore`.
#
# To create the vector store that contains documents relating to CRISPR therapy,
# run the following command:
#
# ```bash
# union run --remote --copy-all agentic_rag.py create_vector_store --query "CRISPR therapy" --load_max_docs 10
# ```
#
# This will get `10` documents from pubmed matching the `"CRISPR therapy"` query.

AgenticRagVectorStore = Artifact(name="agentic-rag-vector-store")


@task(
    container_image=image,
    cache=True,
    cache_version="1",
    secret_requests=[Secret(key="openai_api_key")],
)
@use_pysqlite3
@openai_env_secret
def create_vector_store(
    query: str,
    load_max_docs: Optional[int] = None,
) -> Annotated[FlyteDirectory, AgenticRagVectorStore]:
    """Create a vector store of pubmed documents based on a query."""

    from langchain_community.document_loaders import PubMedLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    load_max_docs = load_max_docs or 10

    loader = PubMedLoader(query, load_max_docs=load_max_docs)
    docs = []
    for doc in loader.load():
        # make sure the title is a string
        title = doc.metadata["Title"]
        if isinstance(title, dict):
            title = " ".join(title.values())
        doc.metadata["Title"] = title
        docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    # Add to vectorDB
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db",
    )
    return FlyteDirectory(path=vector_store._persist_directory)


# Below we define a utility function that reconstitutes the Chroma vector store
# as a Langchain retriever tool, which we'll use in the RAG workflow nodes
# defined below:


def get_vector_store_retriever(path: str):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools.retriever import create_retriever_tool

    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory=path,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_pubmed_research",
        "Search and return information about pubmed research papers relating "
        "to the user query.",
    )
    return retriever_tool


# ## Agent actions and state
#
# Next, we model the agent's actions and state as `Enum`s and `dataclass`es.
# `AgentAction` defines the possible actions the agent can take, which is to
# use the retrieval tool or end the loop. `GraderAction` defines the actions
# resulting from grading the retrieved documents, which can either be to generate
# the final answer or rewrite the user's query.


class AgentAction(Enum):
    tools = "tools"
    end = "end"


class GraderAction(Enum):
    generate = "generate"
    rewrite = "rewrite"
    end = "end"


# We model the `Message`s as a json-encoded string that we can convert to and from
# the Langchain-native messages, and we represent the `AgentState` as a
# list of `Message`s, which will be an append-only list that grows as the RAG
# workflow progresses.


@dataclass
class Message:
    """Json-encoded message."""

    data: str

    def to_langchain(self):
        from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

        data = json.loads(self.data)
        message_type = data.get("type", data.get("role"))
        return {
            "ai": AIMessage,
            "tool": ToolMessage,
            "human": HumanMessage,
        }[
            message_type
        ](**data)

    @classmethod
    def from_langchain(cls, message):
        return cls(data=json.dumps(message.dict()))


@dataclass
class AgentState:
    """A list of messages capturing the state of the RAG execution graph."""

    messages: list[Message]

    def to_langchain(self) -> dict:
        return {"messages": [message.to_langchain() for message in self.messages]}

    def append(self, message):
        self.messages.append(Message.from_langchain(message))

    def __getitem__(self, index):
        message: Message = self.messages[index]
        return message.to_langchain()


# ## Defining the RAG nodes
#
# The next step is to define the nodes of the RAG workflow as actor tasks,
# indicated by the `@actor.task` decorator.

# ### The agent decision
#
# The `agent` task below determines the first conditional branch in the workflow,
# which is whether to end the agent loop or call the retrieval tool. This step
# will end the loop in case the query is not appropriate given the available tools.
#
# The `agent` task runs on the `actor` we defined earlier by decorating the
# `agent` function with `@actor`.
#
# ```{note}
# We use also `@use_pysqlite3`, which is a utility function that makes sure that
# a ChromaDB-compatible version of sqlite3 is installed, and `@openai_env_secret`
# to set the `openai_api_key` secret key as the `OPENAI_API_KEY` environment
# variable.
# ```
#
# This task outputs the updated `AgentState` and the next `AgentAction` to take.


@actor.task
@use_pysqlite3
@openai_env_secret
def agent(
    state: AgentState,
    vector_store: FlyteDirectory,
) -> tuple[AgentState, AgentAction]:
    """Invokes the agent to either end the loop or call the retrieval tool."""

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import PromptTemplate

    vector_store.download()
    retriever_tool = get_vector_store_retriever(vector_store.path)

    prompt = PromptTemplate(
        template="""You are an biomedical research assistant that can retrieve
        documents and answer questions based on those documents.

        Here is the user question: {question} \n

        If the question is related to biomedical research, call the relevant
        tool that you have access to. If the question is not related to
        biomedical research, end the loop with a response that the question
        is not relevant.""",
        input_variables=["question"],
    )

    question_message = state[-1]
    assert question_message.type == "human"

    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools([retriever_tool])
    chain = prompt | model
    response = chain.invoke({"question": question_message.content})

    # Get agent's decision to call the retrieval tool or end the loop
    action = AgentAction.end
    if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
        action = AgentAction.tools

    state.append(response)
    return state, action


# ### Retrieving documents
#
# If the agent decided that the next step in the workflow is to retrieve
# documents, the `retrieve` task will be invoked next. This task retrieves
# documents based on the user's query and updates the `AgentState` with
# additional context.


@actor.task
@use_pysqlite3
@openai_env_secret
def retrieve(
    state: AgentState,
    vector_store: FlyteDirectory,
) -> AgentState:
    """Retrieves documents from the vector store."""

    from langchain_core.messages import AIMessage, ToolMessage

    vector_store.download()
    retriever_tool = get_vector_store_retriever(vector_store.path)

    agent_message = state[-1]
    assert isinstance(agent_message, AIMessage)
    assert len(agent_message.tool_calls) == 1

    # invoke the tool to retrieve documents from the vector store
    tool_call = agent_message.tool_calls[0]
    content = retriever_tool.invoke(tool_call["args"])
    response = ToolMessage(content=content, tool_call_id=tool_call["id"])
    state.append(response)
    return state


# ### Grading the retrieved documents
#
# Based on the retrieved documents from the previous step, the `grade` task
# produces a `GraderAction` to determine whether the documents are relevant to
# the query or not. The `GraderAction.generate` value indicates that the
# final answer can be generated, while `GraderAction.rewrite` indicates that
# the user's query should be rewritten to clarify it's semantic meaning.


@actor.task
@openai_env_secret
def grade(state: AgentState) -> GraderAction:
    """Determines whether the retrieved documents are relevant to the question."""

    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    # Restrict the LLM's output to be a binary "yes" or "no"
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved 
        document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the
        user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the
        document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state.to_langchain()["messages"]

    # get the last "human" and "tool" message, which contains the question and
    # retrieval tool context, respectively
    questions = [m for m in messages if m.type == "human"]
    contexts = [m for m in messages if m.type == "tool"]
    question = questions[-1]
    context = contexts[-1]

    scored_result = chain.invoke(
        {
            "question": question.content,
            "context": context.content,
        }
    )
    score = scored_result.binary_score
    return {
        "yes": GraderAction.generate,
        "no": GraderAction.rewrite,
    }[score]


# ### Rewriting the user's query
#
# If the `grade` task returned `GraderAction.rewrite`, then the `rewrite` task
# will be invoked next to update the user's original query based on the contents
# of the `rewrite_prompt` variable.


@actor.task
@openai_env_secret
def rewrite(state: AgentState) -> AgentState:
    """Transform the query to produce a better question."""

    from langchain_core.messages import HumanMessage
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    messages = state.to_langchain()["messages"]

    # get the last "human", which contains the user question
    questions = [m for m in messages if m.type == "human"]
    question = questions[-1].content

    class rewritten_question(BaseModel):
        """Binary score for relevance check."""

        question: str = Field(description="Rewritten question")
        reason: str = Field(description="Reasoning for the rewrite")

    rewrite_prompt = f"""
    Look at the input and try to reason about the underlying semantic
    intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question and provide your reasoning.
    """

    # define model with structured output for the question rewrite
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    rewriter_model = model.with_structured_output(rewritten_question)

    response = rewriter_model.invoke([HumanMessage(content=rewrite_prompt)])
    message = HumanMessage(
        content=response.question,
        response_metadata={"rewrite_reason": response.reason},
    )
    state.append(message)
    return state


# ### Generating the answer
#
# If the `grade` task returned `GraderAction.generate`, then the `generate` task
# will write the final answer to the (potentially rewritten) user query. The
# `return_answer` task will then pull out the last message from the `AgentState`
# to return the final answer as a string.


@actor.task
@openai_env_secret
def generate(state: AgentState) -> AgentState:
    """Generate an answer based on the state."""

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    messages = state.to_langchain()["messages"]

    # get the last "human" and "tool" message, which contains the question and
    # retrieval tool context, respectively
    questions = [m for m in messages if m.type == "human"]
    contexts = [m for m in messages if m.type == "tool"]
    question = questions[-1]
    context = contexts[-1]

    system_message = """
    You are an assistant for question-answering tasks in the biomedical domain.
    Use the following pieces of retrieved context to answer the question. If you
    don't know the answer, just say that you don't know. Make the answer as
    detailed as possible. If the answer contains acronyms, make sure to expand
    them.

    Question: {question}

    Context: {context}

    Answer:
    """

    prompt = ChatPromptTemplate.from_messages([("human", system_message)])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(
        {
            "context": context.content,
            "question": question.content,
        }
    )
    if isinstance(response, str):
        response = AIMessage(response)

    state.append(response)
    return state


@actor.task
def return_answer(state: AgentState) -> str:
    """Finalize the answer to return a string to the user."""

    if len(state.messages) == 1:
        return f"I'm sorry, I don't understand: '{state.messages}'"
    else:
        data = state.messages[-1].to_langchain()
        return data.content


# ## Defining the RAG execution graph
#
# Now that we have all the building blocks in place, we can define the RAG
# workflow as an execution graph with a recursive loop, which fetches documents
# and rewrites the user query until the grader decides to generate the final
# answer.
#
# We can do that by defining `agent_loop` as a dynamic workflow using the
# `@dynamic` decorator. It takes in the current `AgentState`, the `AgentAction`
# produced by the `agent` task, and the `FlyteDirectory` containing the vector
# store of documents.


@dynamic
def agent_loop(
    state: AgentState,
    action: AgentAction,
    vector_store: FlyteDirectory,
    n_rewrites: int,
) -> AgentState:
    """
    The first conditional branch in the RAG workflow. This determines whether
    the agent loop should end or call the retrieval tool for grading.
    """

    if action == AgentAction.end:
        return state
    elif action == AgentAction.tools:
        state = retrieve(state=state, vector_store=vector_store)
        grader_action = grade(state=state)
        return rewrite_or_generate(
            state=state,
            grader_action=grader_action,
            vector_store=vector_store,
            n_rewrites=n_rewrites,
        )
    else:
        raise RuntimeError(f"Invalid action '{action}'")


# If the `agent` task produced `AgentAction.end` as the action, the loop will
# break and the RAG workflow will respond, with with an answer to the user
# query or an acknowledgement that it doesn't know the answer. However, if the
# action is `AgentAction.tools`, the workflow will call the `retrieve` and `grade`
# tasks, then go into the `rewrite_or_generate` workflow.
#
# `rewrite_or_generate` is also a dynamic workflow. If the `GraderAction` produced
# by the `grade` task is `GraderAction.rewrite`, the workflow will call the
# `rewrite` and `agent` tasks, looping back into the `agent_loop` workflow. If
# the grader action is `GraderAction.generate`, then the recursive loop ends
# and the `generate` task is called to produce the final answer.


@dynamic
def rewrite_or_generate(
    state: AgentState,
    grader_action: GraderAction,
    vector_store: FlyteDirectory,
    n_rewrites: int,
) -> AgentState:
    """
    The second conditional branch in the RAG workflow. This determines whether
    the rewrite the original user's query or generate the final answer.
    """
    if grader_action == GraderAction.generate or n_rewrites >= MAX_REWRITES:
        return generate(state=state)
    elif grader_action == GraderAction.rewrite:
        state = rewrite(state=state)
        state, action = agent(state=state, vector_store=vector_store)
        n_rewrites += 1
        return agent_loop(
            state=state,
            action=action,
            vector_store=vector_store,
            n_rewrites=n_rewrites,
        )
    else:
        raise RuntimeError(f"Invalid action '{grader_action}'")


#
# ```{note}
# The `agent_loop` and `rewrite_or_generate` dynamic workflows uses the
# `n_rewrites` parameter to keep track of the number of times a query has been
# rewritten. To limit the number of recursive calls, we define a global variable
# `MAX_REWRITES` to set avoid Union's recursion limit.
# ````
#
# Finally, we wrap all of this logic into the `pubmed_rag` workflow and define
# an `init_state` task to kick-off the recursive loop with the initial user
# query and `agent`` decision.


@actor.task(cache=True, cache_version="0")
def init_state(user_message: str) -> AgentState:
    """Initialize the AgentState with the user's message."""
    from langchain_core.messages import HumanMessage

    return AgentState(messages=[Message.from_langchain(HumanMessage(user_message))])


@workflow
def agentic_rag_workflow(
    user_message: str,
    vector_store: FlyteDirectory = AgenticRagVectorStore.query(),
) -> str:
    """An agentic retrieval augmented generation workflow."""
    state = init_state(user_message=user_message)
    state, action = agent(state=state, vector_store=vector_store)
    state = agent_loop(
        state=state,
        action=action,
        vector_store=vector_store,
        n_rewrites=0,
    )
    return return_answer(state=state)


# Now you can run the entire workflow with:
#
# ```bash
# union run --remote --copy-all agentic_rag.py agentic_rag_workflow --user_message "Tell me about the latest CRISPR therapies"
# ```
#
# ## Building different assistants
#
# The are many ways that you can modify this example to fulfill a different use case.
# For example, you can change the `query` and `load_max_docs` parameters that
# you pass into `create_vector_store` to fetch a diferent set of documents
# from pubmed.
#
# You can also:
# - Modify the `create_vector_store` task to use a different loader, e.g. `WebBaseLoader`
#   to fetch documents from web pages.
# - Do prompt engineering on the prompts in the `agent`, `grade`, `rewrite`, and
#   `generate` tasks to change the RAG workflow's behavior.
# - Expand the number of tools beyond just the `retriever_tool`.
# - Add more `GraderAction` values to perform more complex actions beyond
#   rewriting the question or generating the answer.
