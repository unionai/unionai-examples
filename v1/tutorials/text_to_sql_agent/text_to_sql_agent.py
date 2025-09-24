# # Natural Language SQL Query Agent using Smolagent
#
# In this tutorial, we build a Union workflow that queries a SQLite database using natural
# language with Smolagent. Given that Agent workflow has high latency, we also define
# a FastAPI app that contains two endpoints hosted on Union's App serving:
# 1. One endpoint to trigger text to SQL query workflow.
# 2. Another endpoint to check the status of the workflow.
#
# {{run-on-union}}
#
# ## Defining the Agent workflow
#
# First, we define the modules and libraries required by the workflow and FastAPI App:

from typing import Annotated
from pathlib import Path
from union import (
    task,
    Resources,
    ImageSpec,
    workflow,
    ActorEnvironment,
    actor_cache,
    Artifact,
    FlyteDirectory,
    current_context,
    Cache,
    UnionRemote,
    Secret,
)
from union.app import App
import os
from pydantic import BaseModel
from fastapi import HTTPException, Security, status, Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from flytekit.extras.accelerators import L4


# For this simple example, we will define a tool that queries a in-memory SQLite database.
# The coding agent will load a large language model with VLLM to translate natural language
# into Python code. We define a `sql_engine` tool so the coding agent can run SQL queries
# on the database.
#
# Given that the large language model takes a long time to load, we use
# Union Actors to keep the pod up and the LLM model loaded. With `actor_cache`, we keep
# the `smolagent`'s `CodeAgent` object, so we do not need to load the LLM again after
# the first workflow execution.
@actor_cache
def get_agent(model: FlyteDirectory):
    """Creates a smolagent CodeAgent."""
    model_path = model.download()

    from smolagents import tool
    from smolagents import CodeAgent, VLLMModel
    from sqlalchemy import (
        create_engine,
        MetaData,
        Table,
        Column,
        String,
        Integer,
        Float,
        insert,
        text,
    )

    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    def insert_rows_into_table(rows, table, engine=engine):
        for row in rows:
            stmt = insert(table).values(**row)
            with engine.begin() as connection:
                connection.execute(stmt)

    table_name = "receipts"
    receipts = Table(
        table_name,
        metadata_obj,
        Column("receipt_id", Integer, primary_key=True),
        Column("customer_name", String(16), primary_key=True),
        Column("price", Float),
        Column("tip", Float),
    )
    metadata_obj.create_all(engine)

    rows = [
        {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
        {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
        {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
        {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
    ]
    insert_rows_into_table(rows, receipts)

    @tool
    def sql_engine(query: str) -> str:
        """
        Allows you to perform SQL queries on the table. Returns a string representation of the result.
        The table is named 'receipts'. Its description is as follows:
            Columns:
            - receipt_id: INTEGER
            - customer_name: VARCHAR(16)
            - price: FLOAT
            - tip: FLOAT

        Args:
            query: The query to perform. This should be correct SQL.
        """
        output = ""
        with engine.connect() as con:
            rows = con.execute(text(query))
            for row in rows:
                output += "\n" + str(row)
        return output

    llm_model = VLLMModel(model_id=model_path)

    agent = CodeAgent(
        tools=[sql_engine],
        model=llm_model,
    )
    return agent


# We define a `ImageSpec` that specifies the packages needed by the text to SQL workflow.
# Be sure to set `REGISTRY` to an image registry that you can push to and the cluster can
# pull from.
image = ImageSpec(
    name="text-to-sql",
    apt_packages=["build-essential"],
    packages=[
        "smolagents[vllm]==1.15.0",
        "union>=0.1.182",
        "sqlalchemy==2.0.40",
        "huggingface-hub[hf_transfer]==0.31.2",
        "hf_xet==1.1.1",
        "fastapi[standard]==0.115.12",
    ],
    registry=os.getenv("REGISTRY", "ghcr.io/unionai-oss"),
    env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)

# In the text to SQL workflow, we define a task to download and cache the `Qwen2.5-Coder-7B-Instruct`
# model from HuggingFace into a Union Artifact. After the first workflow execution, the model weights are stored
# in your blob store, which means the model will load faster in sequential runs.
Qwen_Coder_Artifact = Artifact(name="Qwen2.5-Coder-7B-Instruct")
COMMIT = "c03e6d358207e414f1eca0bb1891e29f1db0e242"


@task(
    requests=Resources(cpu="3", mem="10Gi"),
    limits=Resources(cpu="3", mem="10Gi"),
    cache=Cache(version="v1"),
    container_image=image,
)
def download_model() -> Annotated[FlyteDirectory, Qwen_Coder_Artifact]:
    """Download model from huggingface."""
    from huggingface_hub import snapshot_download

    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    ctx = current_context()
    working_dir = Path(ctx.working_directory)
    hf_cache = working_dir / "cache"
    hf_cache.mkdir()

    snapshot_download(model_id, cache_dir=hf_cache, revision=COMMIT)
    snapshot_dir = hf_cache / "models--Qwen--Qwen2.5-Coder-7B-Instruct" / "snapshots" / COMMIT
    assert snapshot_dir.exists()
    return Qwen_Coder_Artifact.create_from(snapshot_dir)


# We define an `ActorEnvironment` to run the coding agent using L4 resources and enough ephemeral storage
# to pull the large language model into desk.
actor_env = ActorEnvironment(
    name="text-to-sql-actor",
    container_image=image,
    ttl_seconds=100,
    requests=Resources(cpu="3", mem="10Gi", gpu="1", ephemeral_storage="40Gi"),
    limits=Resources(cpu="3", mem="10Gi", gpu="1", ephemeral_storage="40Gi"),
    accelerator=L4,
)


# The `ask` task calls `get_agent` to create the coding agent, passes in the query into the agent.
@actor_env.task()
def ask(
    model: FlyteDirectory,
    query: str,
) -> str:
    agent = get_agent(model)
    result = agent.run(query)
    return str(result)


# Finally, we define the workflow that downloads the model and runs the query.
@workflow
def ask_wf(
    query: str = "Can you give me the name of the client who got the most expensive receipt?",
) -> str:
    model = download_model()
    return ask(model=model, query=query)


# ## Defining the FastAPI App
#
# We use a workflow to define the agent query because it has high latency. For applications, we define
# a simple FastAPI app to interact with two endpoints to interact with the agent:
# 1. One endpoint to trigger text to SQL query workflow.
# 2. Another endpoint to check the status of the workflow.
#
# We start by defining the ImageSpec with the dependencies require by the FastAPI app:

fastapi_image = ImageSpec(
    name="text-to-sql",
    packages=["fastapi[standard]==0.115.12", "union>=0.1.182", "union-runtime>=0.1.18"],
    registry=os.getenv("IMAGE_SPEC_REGISTRY"),
)

# Then we define the FastAPI app that exposes an endpoint to launch and query for the status
# of the agent workflow. The Union `App` adds two secrets:
# 1. `WEBHOOK_API_KEY`: Used to authenticate the FastAPI app
# 2. `MY_UNION_API_KEY`: API key to allow access to the cluster from the FastAPI app.

fastapi = FastAPI()
default_project = os.getenv("UNION_CURRENT_PROJECT", "flytesnacks")
app = App(
    name="text-to-sql-fast-api",
    container_image=fastapi_image,
    framework_app=fastapi,
    secrets=[
        Secret(key="MY_UNION_API_KEY", env_var="UNION_API_KEY"),
        Secret(key="WEBHOOK_API_KEY", env_var="WEBHOOK_API_KEY"),
    ],
    requests=Resources(cpu=1, mem="1Gi"),
    requires_auth=False,
    env={"UNION_CURRENT_PROJECT": default_project},
)


# ### Create the secrets
#
# With `requires_auth=False`, the endpoint can be reached without going through Union's
# authentication, which is okay since we are rolling our own `WEBHOOK_API_KEY`. Before
# we can deploy the app, we create the secrets required by the application:
#
# ```shell
# $ union create secret --name WEBHOOK_API_KEY
# ````
#
# For this example, we'll assume that `WEBHOOK_API_KEY` is defined in your shell.
# Next, to create the `MY_UNION_API_KEY` secret, we need to first create a admin api-key:
#
# ```shell
# $ union create api-key admin --name admin-union-api-key
# ```
#
# You will see a `export UNION_API_KEY=<api-key>`, copy the api key and create a secret
# with it:
#
# ```shell
# $ union create secret --name MY_UNION_API_KEY
# ```


# ## Define the FastAPI Endpoints
#
# The inputs and outputs of the FastAPI endpoints are defined as Pydanatic models:
class QueryInput(BaseModel):
    """Text to translate to SQL"""

    content: str


class ExecutionPromiseOutput(BaseModel):
    """Refers to a running execution."""

    name: str
    is_done: bool
    execution_url: str


class Result(BaseModel):
    """Contains the result of the execution"""

    name: str
    is_done: bool
    output: str


# Here we define `verify_token` and pull in `WEBHOOK_API_KEY`, so that our FastAPI App can authenticate the query.

WEBHOOK_API_KEY = os.getenv("WEBHOOK_API_KEY")
security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> HTTPAuthorizationCredentials:
    if credentials.credentials != WEBHOOK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return credentials


# `run_ask_query` uses `UnionRemote` to launch an execution and return the status of the execution.
@fastapi.post("/ask-query")
def run_ask_query(
    query: QueryInput,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
) -> ExecutionPromiseOutput:
    remote = UnionRemote()
    wf = remote.fetch_workflow(name="text_to_sql_agent.ask_wf", domain="development")
    execution = remote.execute(
        wf,
        inputs={"query": query.content},
        domain="development",
    )
    return ExecutionPromiseOutput(
        name=execution.id.name,
        is_done=execution.is_done,
        execution_url=remote.generate_console_url(execution),
    )


# `check_result` uses `UnionRemote` to check the status of an execution. If the execution is done,
# the output is returned as a `Result`.
@fastapi.get("/ask-query-result/{name}")
def check_result(
    name: str,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(verify_token)],
) -> Result:
    remote = UnionRemote()
    execution = remote.fetch_execution(name=name, domain="development")

    if execution.is_successful:
        output = execution.outputs["o0"]
    else:
        output = ""

    return Result(name=name, is_done=execution.is_done, output=output)


# ## Registering and deploying the application
#
# To register the text-to-sql workflow:
#
# ```shell
# $ union register text_to_sql_agent.py
# ```
#
# To deploy the FastAPI App
#
# ```shell
# $ union deploy apps text_to_sql_agent.py text-to-sql-fast-api
# ```
#
# To launch the text to SQL workflow with the FastAPI frontend, run the following with your `WEBHOOK_API_KEY`:
#
# ```shell
# $curl -X 'POST' \
#  'https://<union-tenant>/ask-query' \
#  -H 'accept: application/json' \
#  -H 'Authorization: Bearer <WEBHOOK_API_KEY>' \
#  -H 'Content-Type: application/json' \
#  -d '{
#  "content": "Can you give me the name of the client who got the most expensive receipt?"
# }'
# ````
#
# This will return a response with the execution name and the URL:
#
# ```json
# {
#   "name": "as7zpkwjk5rccdfhmmz9",
#   "is_done": false,
#   "execution_url": "https://<union-tenant>/console/projects/thomasjpfan/domains/development/executions/as7zpkwjk5rccdfhmmz9"
# }
# ```
#
# To check the status of the query:
#
# ```shell
# $ curl -X 'GET' \
#  'https://<union-tenant>/ask-query-result/as7zpkwjk5rccdfhmmz9' \
#  -H 'accept: application/json' \
#  -H 'Authorization: Bearer <WEBHOOK_API_KEY>'
#
# While the the execution is still running, then you'll get status where "is_done" is false:
#
# ```json
# {
#  "name": "as7zpkwjk5rccdfhmmz9",
#  "is_done": false,
#  "output": ""
# }
# ```

# If the execution is complete, this will return the output of the query:
#
# ```json
# {
#  "name": "as7zpkwjk5rccdfhmmz9",
#  "is_done": true,
#  "output": "Woodrow Wilson"
# }
# ```
