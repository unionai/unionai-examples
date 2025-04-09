# Integrating Arize with Union Serving

This directory contains examples for integrating Arize and Phoenix (open-source) tracing and evaluation into your Union serving apps.

## Tracing

Tracing captures the flow of a request as it passes through various components of an LLM or RAG app — helping you debug and monitor behavior.

### Model tracing (Phoenix)

To deploy a model with Phoenix tracing enabled:

```bash
union create secret phoenix-api-key
union deploy apps apps.py vllm-deepseek
union deploy apps apps.py vllm-deepseek-gradio-phoenix
```

### RAG app with tracing (Arize)

To ingest documents into a vector database:

```bash
union run --remote ingestion.py ingest_docs_workflow --file_path "<YOUR_FILE>"
```

To deploy the RAG app with Arize tracing enabled:

```bash
union create secret arize-api-key
union deploy apps apps.py rag-fastapi-arize
```

## Evaluations

You'll also find examples of LLM-as-a-judge evaluations for RAG systems.

## Arize online evaluation

Register the workflow and activate the launch plan to run it every 5 minutes:

```bash
union create secret arize-api-key

# initial backfill
union run --remote evaluation.py arize_online_evaluation \
    --arize_space_id "<YOUR_SPACE_ID>" \
    --arize_model_id "<YOUR_MODEL_ID>" \
    --arize_project_name "<YOUR_PROJECT_NAME>" \
    --backfill_from_datetime "2025-01-03T01:10:26.249+00:00" \
    --backfill_end_datetime "2025-01-15T01:19:27.249+00:00"

union register evaluation.py arize_online_evaluation

# generate evals every 5 minutes
# before registering, add your arize space ID and model ID to the `evaluation.py` file
union register evaluation.py arize_online_evaluation_lp
```

## Phoenix online evaluation

Register the workflow and activate the launch plan to run it every 5 minutes:

```bash
union create secret phoenix-api-key

# initial backfill
union run --remote evaluation.py phoenix_online_evaluation \
    --backfill_from_datetime "2025-01-03T01:10:26.249+00:00" \
    --backfill_end_datetime "2025-01-15T01:19:27.249+00:00"

union register evaluation.py phoenix_online_evaluation

# generate evals every 5 minutes
union register evaluation.py phoenix_online_evaluation_lp
```
