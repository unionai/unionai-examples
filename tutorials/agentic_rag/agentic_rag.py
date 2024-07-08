import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Annotated

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from flytekit import current_context, dynamic, task, workflow, Artifact, Secret
from flytekit import ImageSpec
from flytekit.types.directory import FlyteDirectory


image = ImageSpec(
    python_version="3.11",
    packages=[
        "beautifulsoup4",
        "chromadb",
        "flytekitplugins-pydantic",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "langchain-text-splitters",
        "langchainhub",
        "langgraph",
        "pysqlite3-binary",
        "tiktoken",
    ]
)


AgenticRagVectorStore = Artifact(name="agentic-rag-vector-store")


class AgentAction(Enum):
    tools = "tools"
    end = "end"


class GraderAction(Enum):
    generate = "generate"
    rewrite = "rewrite"


@dataclass
class AgentState:
    messages: list[str]


def state_to_langchain(state: AgentState) -> dict:
    return {
        "messages": [
            message_to_langchain(message) for message in state.messages
        ]
    }


def message_to_langchain(data: str) -> BaseMessage:
    data = json.loads(data)
    message_type = data.get("type", data.get("role"))
    return {
        "ai": AIMessage,
        "tool": ToolMessage,
        "human": HumanMessage,
    }[message_type](**data)


def env_secret(fn=None, *, secret_name: str, env_var: str):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        os.environ[env_var] = current_context().secrets.get(key=secret_name)
        return fn(*args, **kwargs)
    
    if fn is None:
        return partial(env_secret, secret_name=secret_name, env_var=env_var)

    return wrapper


def use_pysqlite3(fn):
    # workaround for sqlite3 import error

    @wraps(fn)
    def wrapper(*args, **kwargs):
        __import__('pysqlite3')

        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        return fn(*args, **kwargs)
    
    return wrapper


@task(
    container_image=image,
    cache=True,
    cache_version="1",
    secret_requests=[Secret(key="openai_api_key")],
)
@use_pysqlite3
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def create_vector_store() -> Annotated[FlyteDirectory, AgenticRagVectorStore]:
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db",
    )
    return FlyteDirectory(path=vector_store._persist_directory)


@task(container_image=image)
def init_state(user_message: str) -> AgentState:
    return AgentState(
        messages=[json.dumps(HumanMessage(user_message).dict())]
    )


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
@use_pysqlite3
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def agent(
    state: AgentState,
    vector_store: FlyteDirectory,
) -> AgentState:
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """

    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools.retriever import create_retriever_tool

    vector_store.download()
    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory=vector_store.path,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM "
        "agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    state_dict = state_to_langchain(state)
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools([retriever_tool])
    response = model.invoke(state_dict["messages"])
    print("---RESPONSE---")
    print(response)

    state.messages.append(json.dumps(response.dict()))
    return state


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
@use_pysqlite3
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def retrieve(
    state: AgentState,
    vector_store: FlyteDirectory,
) -> AgentState:

    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools.retriever import create_retriever_tool
    from langgraph.prebuilt import ToolNode

    vector_store.download()
    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory=vector_store.path,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM "
        "agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    state_dict = state_to_langchain(state)
    print("---RETRIEVE---")
    tool_node = ToolNode([retriever_tool])
    response = tool_node.invoke(state_dict["messages"])
    assert len(response) == 1
    state.messages.append(json.dumps(response[0].dict()))
    return state


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def grader(state: AgentState) -> GraderAction:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    from langchain_core.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_openai import ChatOpenAI

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with tool and validation
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

    state_dict = state_to_langchain(state)
    messages = state_dict["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    print("---RESPONSE---")
    print(scored_result)
    score = scored_result.binary_score
    return {
        "yes": GraderAction.generate,
        "no": GraderAction.rewrite,
    }[score]


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def rewrite(state: AgentState) -> AgentState:
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    from langchain_openai import ChatOpenAI

    print("---TRANSFORM QUERY---")
    state_dict = state_to_langchain(state)
    messages = state_dict["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / 
    meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    state.messages.append(json.dumps(response.dict()))
    return state


@task(
    container_image=image,
    secret_requests=[Secret(key="openai_api_key")],
)
@env_secret(secret_name="openai_api_key", env_var="OPENAI_API_KEY")
def generate(state: AgentState) -> AgentState:
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    from langchain import hub
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    print("---GENERATE---")
    state_dict = state_to_langchain(state)
    messages = state_dict["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    state.messages.append(json.dumps(response.dict()))
    return state


@task(container_image=image)
def return_answer(state: AgentState) -> str:
    if len(state.messages) == 1:
        return f"I'm sorry, I don't understand: '{state.messages}'"
    else:
        return state.messages[-1]


@dynamic(container_image=image)
def agent_router(
    state: AgentState,
    vector_store: FlyteDirectory,
) -> AgentState:
    from langgraph.prebuilt import tools_condition

    response = state_to_langchain(state)["messages"][-1]
    action_response = tools_condition({"messages": [response]})
    action = {
        "tools": AgentAction.tools,
        "__end__": AgentAction.end,
    }[action_response]

    if action == AgentAction.end:
        return state
    if action == AgentAction.tools:
        state = retrieve(state=state, vector_store=vector_store)
        grader_action = grader(state=state)
        return grader_router(
            state=state,
            grader_action=grader_action,
            vector_store=vector_store,
        )
    else:
        raise RuntimeError(f"Invalid action '{action}'")


@dynamic(container_image=image)
def grader_router(
    state: AgentState,
    grader_action: GraderAction,
    vector_store: FlyteDirectory,
) -> AgentState:
    if grader_action == GraderAction.rewrite:
        state = rewrite(state=state)
        state = agent(state=state, vector_store=vector_store)
        return agent_router(state=state, vector_store=vector_store)
    elif grader_action == GraderAction.generate:
        return generate(state=state)
    else:
        raise RuntimeError(f"Invalid action '{grader_action}'")


@workflow
def main(
    user_message: str,
    vector_store: FlyteDirectory = AgenticRagVectorStore.query(),
) -> str:
    state = init_state(user_message=user_message)
    state = agent(state=state, vector_store=vector_store)
    state = agent_router(state=state, vector_store=vector_store)
    return return_answer(state=state)


@task(container_image=image)
def passthrough(state: AgentState) -> AgentState:
    state.messages.append(json.dumps(AIMessage("passthrough").dict()))
    return state


@dynamic(container_image=image)
def test_router(
    state: AgentState,
) -> AgentState:
    return passthrough(state=state)


@workflow
def test(
    user_message: str,
    vector_store: FlyteDirectory = AgenticRagVectorStore.query(),
) -> AgentState:
    state = init_state(user_message=user_message)
    state = agent(state=state, vector_store=vector_store)
    return retrieve(state=state, vector_store=vector_store)
