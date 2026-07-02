# Code Mode: an LLM writes code, a sandbox runs it, tasks do the work

Chat with a dataset. You ask a question in the browser, the app launches a Flyte
**run** to answer it, and inside that run Claude writes a small Python program that
executes in Flyte's **Monty sandbox**. The sandbox allows no imports, no IO, and no
network, so the model's code can only call the tools you register. The heavy tool, a
DuckDB `query`, is a **durable Flyte task**, so every query the model writes shows up
as a tracked, retryable task you can click into.

## Why "code mode"

Most tool-using agents call tools one at a time: the model asks for a tool, the
result comes back, it reasons, it asks for the next one. For anything multi-step that
is a lot of round-trips. In **code mode** the model instead writes a single program
that orchestrates the tools, with real control flow and composition, and you run that
program once.

Running an LLM's code is normally the scary part. Here it is the safe part: the
program runs in Monty, a restricted interpreter with **no imports, no filesystem, no
network, microsecond startup**. The only things it can do are call the tools you
handed the sandbox.

## How the app runs durable tasks

An app's request handler has **no task context**, so calling a task directly from it
runs the task locally in the app, not on the cluster. To get durable execution the
app **launches the work as a run**. It initializes the client in the app lifespan and
launches a run per request:

```python
@asynccontextmanager
async def lifespan(app):
    # resolves project/domain/endpoint/identity from the pod; org is passed in
    await flyte.init_in_cluster.aio(org=os.getenv("ANALYSIS_ORG") or None)
    yield

# in the request handler:
run = await flyte.run.aio(analyze, message=message)
await run.wait.aio()
result = (await run.outputs.aio())[0]
```

`init_in_cluster` resolves project, domain, endpoint, and identity from the app pod.
The one thing app pods aren't given is the org, so the deploy step reads it from your
config and injects it as `ANALYSIS_ORG`.

Inside that run there *is* a task context, so when the sandboxed code calls `query(...)`
it dispatches as a real child task. The app is a thin front end; the analysis is a
workflow. Each answer comes back with a link to its run in the Union UI. This is the
"call task from app" pattern from the hybrid app-task graphs guide.

## What runs where

| Piece | Where it runs | Why |
|---|---|---|
| the app (`/api/chat`) | a CPU app pod | Serves the UI, launches one analysis run per question. |
| the data preview (`/api/dataset`) | in-process in the app pod | A cheap sample of the rows for demos. No run, no LLM — just shows the data before you query it. |
| `analyze` | a Flyte task (the run) | Owns the code-mode loop: prompt the model, run its code in the sandbox, return the report blocks + summary. |
| `query(sql)` | a **durable child task** | DuckDB over the data. Real work, worth tracking, retrying, and caching. Dispatched from the sandbox. |
| `create_metric`, `create_chart`, `create_table`, `calculate_statistics` | in-process in `analyze` | Microseconds of pure Python; making them tasks would add a round-trip for nothing. |
| the model's code | the Monty sandbox | Untrusted LLM code, confined to calling the tools above. |

`orchestrate_local` classifies each tool it is given, so a durable task and a plain
function can sit side by side in one sandbox call.

## Files

- `tools.py` — the dataset and the tool functions: `query`, plus the report builders `create_metric`, `create_chart`, `create_table`, and `calculate_statistics`.
- `agent.py` — `CodeModeAgent`: builds the system prompt from the tool registry, asks Claude for code, runs it in the sandbox, retries by feeding errors back to the model, and reports which tools the code called.
- `analysis.py` — the durable half: the `analyze` task, the durable `query` task, and the tool registry. Imported by both `app.py` and a standalone `__main__`.
- `app.py` — the FastAPI front end and the launch-a-run wiring. The app declares the run environment via `depends_on`.
- `ui.py` — the single-page chat UI: a **Preview the dataset** button, and renders the report blocks (metrics, charts, tables) plus the strip of tools that ran.

## The data

`tools.py` generates a small `orders` table (ecommerce orders for 2024) so the example
runs with no data to fetch. It is a stand-in for a real table. To query your own data,
change `_dataframe()` in `tools.py` to read a file or a warehouse, for example:

```python
import duckdb
duckdb.read_parquet("s3://my-bucket/orders.parquet")   # or a https:// URL, or a warehouse connection
```

The rest of the agent does not change: the model still writes SQL against a table
called `orders`.

## Run it

You need a Union deployment and an Anthropic API key stored as a secret.

```bash
flyte create secret sam_anthropic_api_key <your-anthropic-key>
python app.py
```

That deploys the app and its run environment together and prints the app URL. Open it and,
if you are recording a demo, click **Preview the dataset** first to show the schema and a
sample of rows (a cheap in-process peek, no run). Then ask something like "give me a 2024
revenue overview" or "return rate by channel." Each answer comes back as a short report
(headline numbers, a chart, sometimes a table) plus a one-line summary and a strip of the
tools the code called, with the generated code under a disclosure and a link to the run so
you can see the query tasks it dispatched.

The app launches runs on your behalf, which needs your org. It reads it from your config
at deploy and injects it as `ANALYSIS_ORG`. If you see `WARNING: no org found`, set it
explicitly: `ANALYSIS_ORG=<your-org> python app.py`.

To exercise the analysis half on its own (no app), run it as a durable `flyte.run`:

```bash
python analysis.py     # runs one analysis and prints the run URL
```

## Going further

- **A bigger, real dataset.** Point `query` at a parquet in object storage or a warehouse (see above). Because `query` is a durable task, large or slow queries get retries and caching for free.
- **A model-based tool.** Add a tool that calls another model (an LLM judge, an embedder) and register it like any other. Cheap tools stay in-process; expensive ones become tasks.
- **More tools.** Write a function with a docstring and add it to the registry in `analysis.py`. The agent regenerates its prompt from the signatures, so there is nothing else to wire up.
