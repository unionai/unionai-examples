# Track and Evaluate RAG with Weave

This project showcases a Retrieval-Augmented Generation (RAG) pipeline that uses a self-hosted LLM, a Weaviate vector database, and Weave by Weights & Biases for full observability, debugging, and response validation.

Given an Airbnb listing URL, the app:

- Generates embeddings for the listing,
- Stores them in Weaviate,
- At query time, applies metadata filters to retrieve relevant listings, and
- Generates a grounded, LLM-based response.

All steps are instrumented and evaluated using Weave, which traces the end-to-end pipeline and includes built-in Guardrails for automatic response validation.

## Project Structure

- `ingestion.py`: Defines the data ingestion task: parses listings, generates embeddings, and stores them in Weaviate.
- `apps.py`: Contains the RAG application logic:
  - Defines a self-hosted LLM app to generate answers.
  - Implements the full RAG flow.
  - Uses Weave to trace, evaluate, and debug each component.
  - Includes Weave Guardrails for quality checks:
    - Entity recall (is the response grounded in retrieved context?)
    - Relevancy (is the retrieved context relevant to the question?)

## Execution

Follow the steps below to run the ingestion pipeline, cache your model, and deploy the RAG app with Union and Weave.

### Ingest Airbnb Listing Data

Create the necessary secrets for Weaviate access:

```bash
union create secret weaviate_url
union create secret weaviate_api_key
```

Run the ingestion task:

```bash
union run --remote ingestion.py ingest_data \
  --inside_airbnb_listings_url <URL> \
  --index_name AirbnbListings
```

ℹ️ You can download the data from [Inside Airbnb](https://insideairbnb.com/get-the-data/).

### Cache the LLM Model

Pre-cache the model you'll use in the RAG app with Union Artifacts. Example using Mixtral:

```bash
union cache model-from-hf mistralai/Mixtral-8x7B-v0.1 \
  --hf-token-key <YOUR_HF_TOKEN> \
  --union-api-key <YOUR_UNION_API_KEY> \
  --cpu 4 \
  --mem 15G
```

Replace the model ID if you're using a different LLM.

### Deploy the RAG App

Set up your Weights & Biases API key for Weave tracing:

```bash
union create secret wandb_api_key
```

Deploy the app:

```bash
union deploy apps apps.py weave-rag \
  --index_name AirbnbListings \
  --model <YOUR_MODEL_ARTIFACT_ID>
```

Replace `<YOUR_MODEL_ARTIFACT_ID>` with the artifact ID returned from the union cache step.

## Example Queries

Note: The RAG app's performance can be improved by using more refined metadata filters.

- Looking for a superhost in Amsterdam with at least 100 reviews and a rating over 4.5.
- I want a quiet place under $120 in Centrum-Oost that allows instant booking.
