import asyncio
import fnmatch
import os
import re
import zipfile

import flyte
import pandas as pd
import requests
from flyte.io import Dir, File
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine
from utils import env


# {{docs-fragment table_info}}
class TableInfo(BaseModel):
    """Information regarding a structured table."""

    table_name: str = Field(..., description="table name (underscores only, no spaces)")
    table_summary: str = Field(
        ..., description="short, concise summary/caption of the table"
    )


# {{/docs-fragment table_info}}


@env.task
async def download_and_extract(zip_path: str, search_glob: str) -> Dir:
    """Download and extract the dataset zip file if not already available."""
    output_zip = "data.zip"
    extract_dir = "wiki_table_questions"

    if not os.path.exists(zip_path):
        response = requests.get(zip_path, stream=True)
        with open(output_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        output_zip = zip_path
        print(f"Using existing file {output_zip}")

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        for member in zip_ref.namelist():
            if fnmatch.fnmatch(member, search_glob):
                zip_ref.extract(member, extract_dir)

    remote_dir = await Dir.from_local(extract_dir)
    return remote_dir


async def read_csv_file(
    csv_file: File, nrows: int | None = None
) -> pd.DataFrame | None:
    """Safely download and parse a CSV file into a DataFrame."""
    try:
        local_csv_file = await csv_file.download()
        return pd.read_csv(local_csv_file, nrows=nrows)
    except Exception as e:
        print(f"Error parsing {csv_file.path}: {e}")
        return None


def sanitize_column_name(col_name: str) -> str:
    """Sanitize column names by replacing spaces/special chars with underscores."""
    return re.sub(r"\W+", "_", col_name)


async def create_table_from_dataframe(
    df: pd.DataFrame, table_name: str, engine, metadata_obj
):
    """Create a SQL table from a Pandas DataFrame."""
    # Sanitize column names
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)

    # Define table columns based on DataFrame dtypes
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]

    table = Table(table_name, metadata_obj, *columns)

    # Create table in database
    metadata_obj.create_all(engine)

    # Insert data into table
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(table.insert().values(**row.to_dict()))


@flyte.trace
async def create_table(
    csv_file: File, table_info: TableInfo, database_path: str
) -> str:
    """Safely create a table from CSV if parsing succeeds."""
    df = await read_csv_file(csv_file)
    if df is None:
        return "false"

    print(f"Creating table: {table_info.table_name}")

    engine = create_engine(f"sqlite:///{database_path}")
    metadata_obj = MetaData()

    await create_table_from_dataframe(df, table_info.table_name, engine, metadata_obj)
    return "true"


@flyte.trace
async def llm_structured_predict(
    df_str: str,
    table_names: list[str],
    prompt_tmpl: ChatPromptTemplate,
    feedback: str,
    llm: OpenAI,
) -> TableInfo:
    return llm.structured_predict(
        TableInfo,
        prompt_tmpl,
        feedback=feedback,
        table_str=df_str,
        exclude_table_name_list=str(list(table_names)),
    )


async def generate_unique_table_info(
    df_str: str,
    table_names: list[str],
    prompt_tmpl: ChatPromptTemplate,
    llm: OpenAI,
    tablename_lock: asyncio.Lock,
    retries: int = 3,
) -> TableInfo | None:
    """Process a single CSV file to generate a unique TableInfo."""
    last_table_name = None
    for attempt in range(retries):
        feedback = ""
        if attempt > 0:
            feedback = f"Note: '{last_table_name}' already exists. Please pick a new name not in {table_names}."

        table_info = await llm_structured_predict(
            df_str, table_names, prompt_tmpl, feedback, llm
        )
        last_table_name = table_info.table_name

        async with tablename_lock:
            if table_info.table_name not in table_names:
                table_names.append(table_info.table_name)
                return table_info

        print(f"Table name {table_info.table_name} already exists, retrying...")

    return None


async def process_csv_file(
    csv_file: File,
    table_names: list[str],
    semaphore: asyncio.Semaphore,
    tablename_lock: asyncio.Lock,
    llm: OpenAI,
    prompt_tmpl: ChatPromptTemplate,
) -> TableInfo | None:
    """Process a single CSV file to generate a unique TableInfo."""
    async with semaphore:
        df = await read_csv_file(csv_file, nrows=10)
        if df is None:
            return None
        return await generate_unique_table_info(
            df.to_csv(), table_names, prompt_tmpl, llm, tablename_lock
        )


@env.task
async def extract_table_info(
    data_dir: Dir, model: str, concurrency: int
) -> list[TableInfo | None]:
    """Extract structured table information from CSV files."""
    table_names: list[str] = []
    semaphore = asyncio.Semaphore(concurrency)
    tablename_lock = asyncio.Lock()
    llm = OpenAI(model=model)

    prompt_str = """\
    Provide a JSON object with the following fields:

    - `table_name`: must be unique and descriptive (underscores only, no generic names).
    - `table_summary`: short and concise summary of the table.

    Do NOT use any of these table names: {exclude_table_name_list}

    Table:
    {table_str}

    {feedback}
    """
    prompt_tmpl = ChatPromptTemplate(
        message_templates=[ChatMessage.from_str(prompt_str, role="user")]
    )

    tasks = [
        process_csv_file(
            csv_file, table_names, semaphore, tablename_lock, llm, prompt_tmpl
        )
        async for csv_file in data_dir.walk()
    ]

    return await asyncio.gather(*tasks)


# {{docs-fragment data_ingestion}}
@env.task
async def data_ingestion(
    csv_zip_path: str = "https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip",
    search_glob: str = "WikiTableQuestions/csv/200-csv/*.csv",
    concurrency: int = 5,
    model: str = "gpt-4o-mini",
) -> tuple[File, list[TableInfo | None]]:
    """Main data ingestion pipeline: download → extract → analyze → create DB."""
    data_dir = await download_and_extract(csv_zip_path, search_glob)
    table_infos = await extract_table_info(data_dir, model, concurrency)

    database_path = "wiki_table_questions.db"

    i = 0
    async for csv_file in data_dir.walk():
        table_info = table_infos[i]
        if table_info:
            ok = await create_table(csv_file, table_info, database_path)
            if ok == "false":
                table_infos[i] = None
        else:
            print(f"Skipping table creation for {csv_file} due to missing TableInfo.")
        i += 1

    db_file = await File.from_local(database_path)
    return db_file, table_infos


# {{/docs-fragment data_ingestion}}
