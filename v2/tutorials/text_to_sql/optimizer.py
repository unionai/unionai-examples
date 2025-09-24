import asyncio
import html
import os
import re
from dataclasses import dataclass
from typing import Optional, Union

import flyte
import flyte.report
import pandas as pd
from data_ingestion import TableInfo
from flyte.io import Dir, File
from llama_index.core import SQLDatabase
from llama_index.core.retrievers import SQLRetriever
from sqlalchemy import create_engine
from text_to_sql import data_ingestion, generate_sql, index_all_tables, retrieve_tables
from utils import env

CSS = """
<style>
    body {
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    }
    .results-table {
        border-collapse: collapse;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        font-size: 14px;
    }
    .results-table th {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 10px;
        text-align: left;
    }
    .results-table td {
        border: 1px solid #ddd;
        padding: 8px;
        vertical-align: top;
    }
    .results-table tr:nth-child(even) {background-color: #f9f9f9;}
    .results-table tr:hover {background-color: #f1f1f1;}
    .correct {color: #2E7D32; font-weight: bold;}
    .incorrect {color: #C62828; font-weight: bold;}
    .summary-card {
        background: #f9fbfd;
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        max-width: 800px;
        margin-top: 12px;
    }
    .summary-card h3 {
        margin-top: 0;
        color: #1e88e5;
        font-size: 16px;
    }
</style>
"""


@env.task
async def data_prep(csv_file: File | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Q&A data from a public Google Sheet CSV export URL and split into val/test DataFrames.
    The sheet should have columns: 'input' and 'target'.
    """
    df = pd.read_csv(
        await csv_file.download() if isinstance(csv_file, File) else csv_file
    )

    if "input" not in df.columns or "target" not in df.columns:
        raise ValueError("Sheet must contain 'input' and 'target' columns.")

    # Shuffle rows
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)

    # Val/Test split
    df_renamed = df.rename(columns={"input": "question", "target": "answer"})

    n = len(df_renamed)
    split = n // 2

    df_val = df_renamed.iloc[:split]
    df_test = df_renamed.iloc[split:]

    return df_val, df_test


@dataclass
class ModelConfig:
    model_name: str
    hosted_model_uri: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = 1000
    timeout: int = 600
    prompt: str = ""


@flyte.trace
async def call_model(
    model_config: ModelConfig,
    messages: list[dict[str, str]],
) -> str:
    from litellm import acompletion

    response = await acompletion(
        model=model_config.model_name,
        api_base=model_config.hosted_model_uri,
        messages=messages,
        temperature=model_config.temperature,
        timeout=model_config.timeout,
        max_tokens=model_config.max_tokens,
    )
    return response.choices[0].message["content"]


@flyte.trace
async def generate_response(db_file: File, sql: str) -> str:
    await db_file.download(local_path="local_db.sqlite")

    engine = create_engine("sqlite:///local_db.sqlite")
    sql_database = SQLDatabase(engine)
    sql_retriever = SQLRetriever(sql_database)

    retrieved_rows = sql_retriever.retrieve(sql)

    if retrieved_rows:
        # Get the structured result and stringify
        return str(retrieved_rows[0].node.metadata["result"])

    return ""


async def generate_and_review(
    index: int,
    question: str,
    answer: str,
    target_model_config: ModelConfig,
    review_model_config: ModelConfig,
    db_file: File,
    table_infos: list[TableInfo | None],
    vector_index_dir: Dir,
) -> dict:
    # Generate response from target model
    table_context = await retrieve_tables(
        question, table_infos, db_file, vector_index_dir
    )
    sql = await generate_sql(
        question,
        table_context,
        target_model_config.model_name,
        target_model_config.prompt,
    )
    sql = sql.replace("sql\n", "")

    try:
        response = await generate_response(db_file, sql)
    except Exception as e:
        print(f"Failed to generate response for question {question}: {e}")
        response = None

    # Format review prompt with response + answer
    review_messages = [
        {
            "role": "system",
            "content": review_model_config.prompt.format(
                query_str=question,
                response=response,
                answer=answer,
            ),
        }
    ]
    verdict = await call_model(review_model_config, review_messages)

    # Normalize verdict
    verdict_clean = verdict.strip().lower()
    if verdict_clean not in {"true", "false"}:
        verdict_clean = "not sure"

    return {
        "index": index,
        "model_response": response,
        "sql": sql,
        "is_correct": verdict_clean == "true",
    }


async def run_grouped_task(
    i,
    index,
    question,
    answer,
    sql,
    semaphore,
    target_model_config,
    review_model_config,
    counter,
    counter_lock,
    db_file,
    table_infos,
    vector_index_dir,
):
    async with semaphore:
        with flyte.group(name=f"row-{i}"):
            result = await generate_and_review(
                index,
                question,
                answer,
                target_model_config,
                review_model_config,
                db_file,
                table_infos,
                vector_index_dir,
            )

            async with counter_lock:
                # Update counters
                counter["processed"] += 1
                if result["is_correct"]:
                    counter["correct"] += 1
                    correct_html = "<span class='correct'>✔ Yes</span>"
                else:
                    correct_html = "<span class='incorrect'>✘ No</span>"

                # Calculate accuracy
                accuracy_pct = (counter["correct"] / counter["processed"]) * 100

            # Update chart
            await flyte.report.log.aio(
                f"<script>updateAccuracy({accuracy_pct});</script>",
                do_flush=True,
            )

            # Add row to table
            await flyte.report.log.aio(
                f"""
                <tr>
                    <td>{html.escape(question)}</td>
                    <td>{html.escape(answer)}</td>
                    <td>{html.escape(sql)}</td>
                    <td>{result['model_response']}</td>
                    <td>{result['sql']}</td>
                    <td>{correct_html}</td>
                </tr>
                """,
                do_flush=True,
            )

            return result


@dataclass
class DatabaseConfig:
    csv_zip_path: str
    search_glob: str
    concurrency: int
    model: str


# {{docs-fragment evaluate_prompt}}
@env.task(report=True)
async def evaluate_prompt(
    df: pd.DataFrame,
    target_model_config: ModelConfig,
    review_model_config: ModelConfig,
    concurrency: int,
    db_config: DatabaseConfig,
) -> float:
    semaphore = asyncio.Semaphore(concurrency)
    counter = {"correct": 0, "processed": 0}
    counter_lock = asyncio.Lock()

    # Write initial HTML structure
    await flyte.report.log.aio(
        CSS
        + """
        <script>
            function updateAccuracy(percent) {
                const bar = document.getElementById('acc-bar');
                const label = document.getElementById('acc-label');
                bar.setAttribute('width', percent * 3);
                label.textContent = `Accuracy: ${percent.toFixed(1)}%`;
            }
        </script>

        <h2 style="margin-top:0;">Model Evaluation Results</h2>
        <h3>Live Accuracy</h3>
        <svg width="320" height="30" id="accuracy-chart">
            <defs>
                <linearGradient id="acc-gradient" x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%" stop-color="#66bb6a"/>
                    <stop offset="100%" stop-color="#2e7d32"/>
                </linearGradient>
            </defs>
            <rect width="300" height="20" fill="#ddd" rx="5" ry="5"></rect>
            <rect id="acc-bar" width="0" height="20" fill="url(#acc-gradient)" rx="5" ry="5"></rect>
            <text id="acc-label" x="150" y="15" font-size="12" font-weight="bold" text-anchor="middle" fill="#000">
                Accuracy: 0.0%
            </text>
        </svg>

        <table class="results-table">
            <thead>
                <tr>
                    <th>Question</th>
                    <th>Ground Truth Answer</th>
                    <th>Ground Truth SQL</th>
                    <th>Model Response</th>
                    <th>Model SQL</th>
                    <th>Correct?</th>
                </tr>
            </thead>
            <tbody>
        """,
        do_flush=True,
    )

    db_file, table_infos = await data_ingestion(
        db_config.csv_zip_path,
        db_config.search_glob,
        db_config.concurrency,
        db_config.model,
    )

    vector_index_dir = await index_all_tables(db_file)

    # Launch tasks concurrently
    tasks = [
        run_grouped_task(
            i,
            row.Index,
            row.question,
            row.answer,
            row.sql,
            semaphore,
            target_model_config,
            review_model_config,
            counter,
            counter_lock,
            db_file,
            table_infos,
            vector_index_dir,
        )
        for i, row in enumerate(df.itertuples(index=True))
    ]
    await asyncio.gather(*tasks)

    # Close table
    await flyte.report.log.aio("</tbody></table>", do_flush=True)

    async with counter_lock:
        return (
            (counter["correct"] / counter["processed"]) if counter["processed"] else 0.0
        )


# {{/docs-fragment evaluate_prompt}}


@dataclass
class PromptResult:
    prompt: str
    accuracy: float


# {{docs-fragment prompt_optimizer}}
@env.task(report=True)
async def prompt_optimizer(
    df_val: pd.DataFrame,
    target_model_config: ModelConfig,
    review_model_config: ModelConfig,
    optimizer_model_config: ModelConfig,
    max_iterations: int,
    concurrency: int,
    db_config: DatabaseConfig,
) -> tuple[str, float]:
    prompt_accuracies: list[PromptResult] = []

    # Send styling + table header immediately
    await flyte.report.log.aio(
        CSS
        + """
    <h2 style="margin-bottom:6px;">📊 Prompt Accuracy Comparison</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Prompt</th>
                <th>Accuracy</th>
            </tr>
        </thead>
    <tbody>
    """,
        do_flush=True,
    )

    # Step 1: Evaluate starting prompt and stream row
    with flyte.group(name="baseline_evaluation"):
        starting_accuracy = await evaluate_prompt(
            df_val,
            target_model_config,
            review_model_config,
            concurrency,
            db_config,
        )
        prompt_accuracies.append(
            PromptResult(prompt=target_model_config.prompt, accuracy=starting_accuracy)
        )

        await _log_prompt_row(target_model_config.prompt, starting_accuracy)

    # Step 2: Optimize prompts one by one, streaming after each
    while len(prompt_accuracies) <= max_iterations:
        with flyte.group(name=f"prompt_optimization_step_{len(prompt_accuracies)}"):
            # Prepare prompt scores string for optimizer
            prompt_scores_str = "\n".join(
                f"{result.prompt}: {result.accuracy:.2f}"
                for result in sorted(prompt_accuracies, key=lambda x: x.accuracy)
            )

            optimizer_model_prompt = optimizer_model_config.prompt.format(
                prompt_scores_str=prompt_scores_str
            )
            response = await call_model(
                optimizer_model_config,
                [{"role": "system", "content": optimizer_model_prompt}],
            )
            response = response.strip()

            match = re.search(r"\[\[(.*?)\]\]", response, re.DOTALL)
            if not match:
                print("No new prompt found. Skipping.")
                continue

            new_prompt = match.group(1)
            target_model_config.prompt = new_prompt
            accuracy = await evaluate_prompt(
                df_val,
                target_model_config,
                review_model_config,
                concurrency,
                db_config,
            )
            prompt_accuracies.append(PromptResult(prompt=new_prompt, accuracy=accuracy))

            # Log this new prompt row immediately
            await _log_prompt_row(new_prompt, accuracy)

    # Close table
    await flyte.report.log.aio("</tbody></table>", do_flush=True)

    # Find best
    best_result = max(prompt_accuracies, key=lambda x: x.accuracy)
    improvement = best_result.accuracy - starting_accuracy

    # Summary
    await flyte.report.log.aio(
        f"""
    <div class="summary-card">
        <h3>🏆 Summary</h3>
        <p><strong>Best Prompt:</strong> {html.escape(best_result.prompt)}</p>
        <p><strong>Best Accuracy:</strong> {best_result.accuracy*100:.2f}%</p>
        <p><strong>Improvement Over Baseline:</strong> {improvement*100:.2f}%</p>
    </div>
    """,
        do_flush=True,
    )

    return best_result.prompt, best_result.accuracy


# {{/docs-fragment prompt_optimizer}}


async def _log_prompt_row(prompt: str, accuracy: float):
    """Helper to log a single prompt/accuracy row to Flyte report."""
    pct = accuracy * 100
    if pct > 80:
        color = "linear-gradient(90deg, #4CAF50, #81C784)"
    elif pct > 60:
        color = "linear-gradient(90deg, #FFC107, #FFD54F)"
    else:
        color = "linear-gradient(90deg, #F44336, #E57373)"

    await flyte.report.log.aio(
        f"""
        <tr>
            <td>{html.escape(prompt)}</td>
            <td>
                {pct:.1f}%
                <div class="accuracy-bar-container">
                    <div class="accuracy-bar" style="width:{pct*1.6}px; background:{color};"></div>
                </div>
            </td>
        </tr>
        """,
        do_flush=True,
    )


# {{docs-fragment auto_prompt_engineering}}
@env.task
async def auto_prompt_engineering(
    ground_truth_csv: File | str = "/root/ground_truth.csv",
    db_config: DatabaseConfig = DatabaseConfig(
        csv_zip_path="https://github.com/ppasupat/WikiTableQuestions/releases/download/v1.0.2/WikiTableQuestions-1.0.2-compact.zip",
        search_glob="WikiTableQuestions/csv/200-csv/*.csv",
        concurrency=5,
        model="gpt-4o-mini",
    ),
    target_model_config: ModelConfig = ModelConfig(
        model_name="gpt-4.1-mini",
        hosted_model_uri=None,
        prompt="""Given an input question, create a syntactically correct {dialect} query to run.

Schema:
{schema}

Question: {query_str}

SQL query to run:
""",
        max_tokens=10000,
    ),
    review_model_config: ModelConfig = ModelConfig(
        model_name="gpt-4.1",
        hosted_model_uri=None,
        prompt="""Your job is to determine whether the model's response is correct compared to the ground truth taking into account the context of the question.
Both answers were generated by running SQL queries on the same database.

- If the model's response contains all of the ground truth values, and any additional information is harmless (e.g., extra columns or metadata), output "True".
- If it adds incorrect or unrelated rows, or omits required values, output "False".

Question:
{query_str}

Ground Truth:
{answer}

Model Response:
{response}
""",
    ),
    optimizer_model_config: ModelConfig = ModelConfig(
        model_name="gpt-4.1",
        hosted_model_uri=None,
        temperature=0.7,
        max_tokens=None,
        prompt="""
<EXPLANATION>
I have some prompts along with their corresponding accuracies.
The prompts are arranged in ascending order based on their accuracy, where higher accuracy indicates better quality.
</EXPLANATION>

<PROMPTS>
{prompt_scores_str}
</PROMPTS>

Each prompt was used to translate a natural-language question into a SQL query against a provided database schema.

<EXAMPLE>
<SCHEMA>
artists(id, name)
albums(id, title, artist_id, release_year)
</SCHEMA>
<QUESTION>
How many albums did The Beatles release?
</QUESTION>
<ANSWER>
SELECT COUNT(*) FROM albums a JOIN artists r ON a.artist_id = r.id WHERE r.name = 'The Beatles';
</ANSWER>
</EXAMPLE>

<TASK>
Write a new prompt that will achieve an accuracy as high as possible and that is different from the old ones.
</TASK>

<RULES>
- It is very important that the new prompt is distinct from ALL the old ones!
- Ensure that you analyse the prompts with a high accuracy and reuse the patterns that worked in the past.
- Ensure that you analyse the prompts with a low accuracy and avoid the patterns that didn't work in the past.
- Think out loud before creating the prompt. Describe what has worked in the past and what hasn't. Only then create the new prompt.
- Use all available information like prompt length, formal/informal use of language, etc. for your analysis.
- Be creative, try out different ways of prompting the model. You may even come up with hypothetical scenarios that might improve the accuracy.
- You are generating a system prompt. Always use three placeholders for each prompt: dialect, schema, query_str.
- Write your new prompt in double square brackets. Use only plain text for the prompt text and do not add any markdown (i.e. no hashtags, backticks, quotes, etc).
</RULES>
""",
    ),
    max_iterations: int = 5,
    concurrency: int = 10,
) -> dict[str, Union[str, float]]:
    if isinstance(ground_truth_csv, str) and os.path.isfile(ground_truth_csv):
        ground_truth_csv = await File.from_local(ground_truth_csv)

    df_val, df_test = await data_prep(ground_truth_csv)

    best_prompt, val_accuracy = await prompt_optimizer(
        df_val,
        target_model_config,
        review_model_config,
        optimizer_model_config,
        max_iterations,
        concurrency,
        db_config,
    )

    with flyte.group(name="test_data_evaluation"):
        baseline_test_accuracy = await evaluate_prompt(
            df_test,
            target_model_config,
            review_model_config,
            concurrency,
            db_config,
        )

        target_model_config.prompt = best_prompt
        test_accuracy = await evaluate_prompt(
            df_test,
            target_model_config,
            review_model_config,
            concurrency,
            db_config,
        )

    return {
        "best_prompt": best_prompt,
        "validation_accuracy": val_accuracy,
        "baseline_test_accuracy": baseline_test_accuracy,
        "test_accuracy": test_accuracy,
    }


# {{/docs-fragment auto_prompt_engineering}}

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(auto_prompt_engineering)
    print(run.url)
