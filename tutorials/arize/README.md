# Integrating Arize with Union Serving

This directory contains examples for integrating Arize and Phoenix (open-source) tracing and evaluation into your Union serving apps.

## Tracing

Tracing captures the flow of a request as it passes through various components of an LLM or RAG app — helping you debug and monitor behavior.

### Cache the LLM Model

Pre-cache the model you'll use in the RAG app with Union Artifacts.

```bash
union cache model-from-hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --hf-token-key <YOUR_HF_TOKEN> \
  --union-api-key <YOUR_UNION_API_KEY> \
  --cpu 4 \
  --mem 15G
```

Replace the model ID if you're using a different LLM.

### Model tracing (Phoenix)

To deploy a model with Phoenix tracing enabled:

```bash
union create secret phoenix-api-key
union deploy apps apps.py vllm-deepseek-gradio-phoenix \
  --model <YOUR_MODEL_ARTIFACT_ID>
```

Replace `<YOUR_MODEL_ARTIFACT_ID>` with the artifact ID returned from the union cache step.

### RAG app with tracing (Arize)

To ingest documents into a vector database:

```bash
union run --remote ingestion.py ingest_docs_workflow --file_path <YOUR_FILE>
```

To deploy the RAG app with Arize tracing enabled:

```bash
union create secret arize-api-key
union deploy apps apps.py rag-fastapi-arize \
  --arize-space-id <YOUR_SPACE_ID> \
  --model <YOUR_MODEL_ARTIFACT_ID>
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
# before registering, add your phoenix project name to the `evaluation.py` file
union register evaluation.py phoenix_online_evaluation_lp
```
