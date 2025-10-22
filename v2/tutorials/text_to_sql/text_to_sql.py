# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
#    "llama-index-core>=0.11.0",
#    "llama-index-llms-openai>=0.2.0",
#    "sqlalchemy>=2.0.0",
#    "pandas>=2.0.0",
#    "requests>=2.25.0",
#    "pydantic>=2.0.0",
# ]
# main = "text_to_sql"
# params = ""
# ///

import asyncio
from pathlib import Path

import flyte
from data_ingestion import TableInfo, data_ingestion
from flyte.io import Dir, File
from llama_index.core import (
    PromptTemplate,
    SQLDatabase,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.llms import ChatResponse
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, text
from utils import env


# {{docs-fragment index_tables}}
@flyte.trace
async def index_table(table_name: str, table_index_dir: str, database_uri: str) -> str:
    """Index a single table into vector store."""
    path = f"{table_index_dir}/{table_name}"
    engine = create_engine(database_uri)

    def _fetch_rows():
        with engine.connect() as conn:
            cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
            return cursor.fetchall()

    result = await asyncio.to_thread(_fetch_rows)
    nodes = [TextNode(text=str(tuple(row))) for row in result]
    index = VectorStoreIndex(nodes)
    index.set_index_id("vector_index")
    index.storage_context.persist(path)

    return path


@env.task
async def index_all_tables(db_file: File) -> Dir:
    """Index all tables concurrently."""
    table_index_dir = "table_indices"
    Path(table_index_dir).mkdir(exist_ok=True)

    await db_file.download(local_path="local_db.sqlite")
    engine = create_engine("sqlite:///local_db.sqlite")
    sql_database = SQLDatabase(engine)

    tasks = [
        index_table(t, table_index_dir, "sqlite:///local_db.sqlite")
        for t in sql_database.get_usable_table_names()
    ]
    await asyncio.gather(*tasks)

    remote_dir = await Dir.from_local(table_index_dir)
    return remote_dir


# {{/docs-fragment index_tables}}


@flyte.trace
async def get_table_schema_context(
    table_schema_obj: SQLTableSchema,
    database_uri: str,
) -> str:
    """Retrieve schema + optional description context for a single table."""
    engine = create_engine(database_uri)
    sql_database = SQLDatabase(engine)

    table_info = sql_database.get_single_table_info(table_schema_obj.table_name)

    if table_schema_obj.context_str:
        table_info += f" The table description is: {table_schema_obj.context_str}"

    return table_info


@flyte.trace
async def get_table_row_context(
    table_schema_obj: SQLTableSchema,
    local_vector_index_dir: str,
    query: str,
) -> str:
    """Retrieve row-level context examples using vector search."""
    storage_context = StorageContext.from_defaults(
        persist_dir=str(f"{local_vector_index_dir}/{table_schema_obj.table_name}")
    )
    vector_index = load_index_from_storage(storage_context, index_id="vector_index")
    vector_retriever = vector_index.as_retriever(similarity_top_k=2)
    relevant_nodes = vector_retriever.retrieve(query)

    if not relevant_nodes:
        return ""

    row_context = "\nHere are some relevant example rows (values in the same order as columns above)\n"
    for node in relevant_nodes:
        row_context += str(node.get_content()) + "\n"

    return row_context


async def process_table(
    table_schema_obj: SQLTableSchema,
    database_uri: str,
    local_vector_index_dir: str,
    query: str,
) -> str:
    """Combine schema + row context for one table."""
    table_info = await get_table_schema_context(table_schema_obj, database_uri)
    row_context = await get_table_row_context(
        table_schema_obj, local_vector_index_dir, query
    )

    full_context = table_info
    if row_context:
        full_context += "\n" + row_context

    print(f"Table Info: {full_context}")
    return full_context


async def get_table_context_and_rows_str(
    query: str,
    database_uri: str,
    table_schema_objs: list[SQLTableSchema],
    vector_index_dir: Dir,
):
    """Get combined schema + row context for all tables."""
    local_vector_index_dir = await vector_index_dir.download()

    # run per-table work concurrently
    context_strs = await asyncio.gather(
        *[
            process_table(t, database_uri, local_vector_index_dir, query)
            for t in table_schema_objs
        ]
    )

    return "\n\n".join(context_strs)


# {{docs-fragment retrieve_tables}}
@env.task
async def retrieve_tables(
    query: str,
    table_infos: list[TableInfo | None],
    db_file: File,
    vector_index_dir: Dir,
) -> str:
    """Retrieve relevant tables and return schema context string."""
    await db_file.download(local_path="local_db.sqlite")
    engine = create_engine("sqlite:///local_db.sqlite")
    sql_database = SQLDatabase(engine)

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
        if t is not None
    ]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    retrieved_schemas = obj_retriever.retrieve(query)
    return await get_table_context_and_rows_str(
        query, "sqlite:///local_db.sqlite", retrieved_schemas, vector_index_dir
    )


# {{/docs-fragment retrieve_tables}}


def parse_response_to_sql(chat_response: ChatResponse) -> str:
    """Extract SQL query from LLM response."""
    response = chat_response.message.content
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    return response.strip().strip("```").strip()


# {{docs-fragment sql_and_response}}
@env.task
async def generate_sql(query: str, table_context: str, model: str, prompt: str) -> str:
    """Generate SQL query from natural language question and table context."""
    llm = OpenAI(model=model)

    fmt_messages = (
        PromptTemplate(
            prompt,
            prompt_type=PromptType.TEXT_TO_SQL,
        )
        .partial_format(dialect="sqlite")
        .format_messages(query_str=query, schema=table_context)
    )

    chat_response = await llm.achat(fmt_messages)
    return parse_response_to_sql(chat_response)


@env.task
async def generate_response(query: str, sql: str, db_file: File, model: str) -> str:
    """Run SQL query on database and synthesize final response."""
    await db_file.download(local_path="local_db.sqlite")

    engine = create_engine("sqlite:///local_db.sqlite")
    sql_database = SQLDatabase(engine)
    sql_retriever = SQLRetriever(sql_database)

    retrieved_rows = sql_retriever.retrieve(sql)

    response_synthesis_prompt = PromptTemplate(
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQL: {sql_query}\n"
        "SQL Response: {context_str}\n"
        "Response: "
    )

    llm = OpenAI(model=model)
    fmt_messages = response_synthesis_prompt.format_messages(
        sql_query=sql,
        context_str=str(retrieved_rows),
        query_str=query,
    )
    chat_response = await llm.achat(fmt_messages)
    return chat_response.message.content


# {{/docs-fragment sql_and_response}}


# {{docs-fragment text_to_sql}}
@env.task
async def text_to_sql(
    system_prompt: str = (
        "Given an input question, first create a syntactically correct {dialect} "
        "query to run, then look at the results of the query and return the answer. "
        "You can order the results by a relevant column to return the most "
        "interesting examples in the database.\n\n"
        "Never query for all the columns from a specific table, only ask for a "
        "few relevant columns given the question.\n\n"
        "Pay attention to use only the column names that you can see in the schema "
        "description. "
        "Be careful to not query for columns that do not exist. "
        "Pay attention to which column is in which table. "
        "Also, qualify column names with the table name when needed. "
        "You are required to use the following format, each taking one line:\n\n"
        "Question: Question here\n"
        "SQLQuery: SQL Query to run\n"
        "SQLResult: Result of the SQLQuery\n"
        "Answer: Final answer here\n\n"
        "Only use tables listed below.\n"
        "{schema}\n\n"
        "Question: {query_str}\n"
        "SQLQuery: "
    ),
    query: str = "What was the year that The Notorious BIG was signed to Bad Boy?",
    model: str = "gpt-4o-mini",
) -> str:
    db_file, table_infos = await data_ingestion()
    vector_index_dir = await index_all_tables(db_file)
    table_context = await retrieve_tables(query, table_infos, db_file, vector_index_dir)
    sql = await generate_sql(query, table_context, model, system_prompt)
    return await generate_response(query, sql, db_file, model)


# {{/docs-fragment text_to_sql}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(text_to_sql)
    print(run.url)
    run.wait()
