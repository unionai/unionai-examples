# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
#    "pandas>=2.0.0",
#    "llama-index-core>=0.11.0",
#    "llama-index-llms-openai>=0.2.0",
#    "pydantic>=2.0.0",
# ]
# main = "build_eval_dataset"
# params = ""
# ///

import sqlite3

import flyte
import pandas as pd
from data_ingestion import data_ingestion
from flyte.io import File
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from utils import env
from pydantic import BaseModel


class QAItem(BaseModel):
    question: str
    sql: str


class QAList(BaseModel):
    items: list[QAItem]


# {{docs-fragment get_and_split_schema}}
@env.task
async def get_and_split_schema(db_file: File, tables_per_chunk: int) -> list[str]:
    """
    Download the SQLite DB, extract schema info (columns + sample rows),
    then split it into chunks with up to `tables_per_chunk` tables each.
    """
    await db_file.download(local_path="local_db.sqlite")
    conn = sqlite3.connect("local_db.sqlite")
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    schema_blocks = []
    for table in tables:
        table_name = table[0]

        # columns
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        block = f"Table: {table_name}({', '.join(columns)})"

        # sample rows
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")
        rows = cursor.fetchall()
        if rows:
            block += "\nSample rows:\n"
            for row in rows:
                block += f"{row}\n"

        schema_blocks.append(block)

    conn.close()

    chunks = []
    current_chunk = []
    for block in schema_blocks:
        current_chunk.append(block)
        if len(current_chunk) >= tables_per_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


# {{/docs-fragment get_and_split_schema}}


# {{docs-fragment generate_questions_and_sql}}
@flyte.trace
async def generate_questions_and_sql(
    schema: str, num_samples: int, batch_size: int
) -> QAList:
    llm = OpenAI(model="gpt-4.1")

    prompt_tmpl = PromptTemplate(
        """Prompt: You are helping build a Text-to-SQL dataset.

Here is the database schema:
{schema}

Generate {num} natural language questions a user might ask about this database.
For each question, also provide the correct SQL query.

Reasoning process (you must follow this internally):

- Given an input question, first create a syntactically correct {dialect} SQL query.
- Never use SELECT *; only include the relevant columns.
- Use only columns/tables from the schema. Qualify column names when ambiguous.
- You may order results by a meaningful column to make the query more useful.
- Be careful not to add unnecessary columns.
- Use filters, aggregations, joins, grouping, and subqueries when relevant.

Final Output:
Return only a JSON object with one field:

- "items": a list of {num} objects, each with:
    - "question": the natural language question
    - "sql": the corresponding SQL query
"""
    )

    all_items: list[QAItem] = []

    # batch generation
    for start in range(0, num_samples, batch_size):
        current_num = min(batch_size, num_samples - start)
        response = llm.structured_predict(
            QAList,
            prompt_tmpl,
            schema=schema,
            num=current_num,
        )
        all_items.extend(response.items)

    # deduplicate
    seen = set()
    unique_items: list[QAItem] = []
    for item in all_items:
        key = (item.question.strip().lower(), item.sql.strip().lower())
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return QAList(items=unique_items[:num_samples])


# {{/docs-fragment generate_questions_and_sql}}


@flyte.trace
async def llm_validate_batch(pairs: list[dict[str, str]]) -> list[str]:
    """Validate a batch of question/sql/result dicts using one LLM call."""
    batch_prompt = """You are validating the correctness of SQL query results against the question.
For each example, answer only "True" (correct) or "False" (incorrect).
Output one answer per line, in the same order as the examples.
---
"""

    for i, pair in enumerate(pairs, start=1):
        batch_prompt += f"""
Example {i}:
Question:
{pair['question']}

SQL:
{pair['sql']}

Result:
{pair['rows']}
---
"""

    llm = OpenAI(model="gpt-4.1")
    resp = await llm.acomplete(batch_prompt)

    # Expect exactly one True/False per example
    results = [
        line.strip()
        for line in resp.text.splitlines()
        if line.strip() in ("True", "False")
    ]
    return results


# {{docs-fragment validate_sql}}
@env.task
async def validate_sql(
    db_file: File, question_sql_pairs: QAList, batch_size: int
) -> list[dict[str, str]]:
    await db_file.download(local_path="local_db.sqlite")
    conn = sqlite3.connect("local_db.sqlite")
    cursor = conn.cursor()

    qa_data = []
    batch = []

    for pair in question_sql_pairs.items:
        q, sql = pair.question, pair.sql
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            batch.append({"question": q, "sql": sql, "rows": str(rows)})

            # process when batch is full
            if len(batch) == batch_size:
                results = await llm_validate_batch(batch)
                for pair, is_valid in zip(batch, results):
                    if is_valid == "True":
                        qa_data.append(
                            {
                                "input": pair["question"],
                                "sql": pair["sql"],
                                "target": pair["rows"],
                            }
                        )
                    else:
                        print(f"Filtered out incorrect result for: {pair['question']}")
                batch = []
        except Exception as e:
            print(f"Skipping invalid SQL: {sql} ({e})")

    # process leftover batch
    if batch:
        results = await llm_validate_batch(batch)
        for pair, is_valid in zip(batch, results):
            if is_valid == "True":
                qa_data.append(
                    {
                        "input": pair["question"],
                        "sql": pair["sql"],
                        "target": pair["rows"],
                    }
                )
            else:
                print(f"Filtered out incorrect result for: {pair['question']}")

    conn.close()
    return qa_data


# {{/docs-fragment validate_sql}}


@flyte.trace
async def save_to_csv(qa_data: list[dict]) -> File:
    df = pd.DataFrame(qa_data, columns=["input", "target", "sql"])

    csv_file = "qa_dataset.csv"
    df.to_csv(csv_file, index=False)

    return await File.from_local(csv_file)


# {{docs-fragment build_eval_dataset}}
@env.task
async def build_eval_dataset(
    num_samples: int = 300, batch_size: int = 30, tables_per_chunk: int = 3
) -> File:
    db_file, _ = await data_ingestion()
    schema_chunks = await get_and_split_schema(db_file, tables_per_chunk)

    per_chunk_samples = max(1, num_samples // len(schema_chunks))
    final_qa_data = []

    for chunk in schema_chunks:
        qa_list = await generate_questions_and_sql(
            schema=chunk,
            num_samples=per_chunk_samples,
            batch_size=batch_size,
        )
        qa_data = await validate_sql(db_file, qa_list, batch_size)
        final_qa_data.extend(qa_data)

    csv_file = await save_to_csv(final_qa_data)
    return csv_file


# {{/docs-fragment build_eval_dataset}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(build_eval_dataset)
    print(run.url)
    run.wait()
