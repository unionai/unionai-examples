# Contextual RAG with Together and Union

## Overview

This project demonstrates a Contextual RAG app that integrates web scraping, embedding generation, and serving using Together.ai and Union.
The workflow includes the following steps:

- Fetches all links to Paul Graham's essays.
- Scrapes the web content to retrieve the essay content.
- Creates text chunks from the scraped content.
- Appends context from the relevant essay to each chunk.
- Generates embeddings and stores them in a vector database.
- Creates a keyword index for efficient retrieval.
- Serves the Chroma vector database for storing embeddings.
- Serves a FastAPI app to expose the RAG functionality.
- Serves a Gradio app that uses the FastAPI endpoint underneath to provide a user interface for the RAG app.

## Tools Used

- Together.ai: Embedding, reranker, and chat APIs.
- Union: Orchestration in a Jupyter notebook, providing the following features:
  - Actors to share the environment between tasks.
  - Map tasks to run tasks in parallel.
  - Secrets management for sensitive data.
  - Artifacts for storing outputs and data lineage.
  - Caching for task optimization.
  - Versioning to track changes.
  - LaunchPlan to schedule and execute the workflow.

## Jupyter Notebook

- Deploy the Chroma database:
  ```
  ENABLE_UNION_SERVING=1 REGISTRY=<YOUR_REGISTRY> union deploy apps -p demo app.py contextual-rag-chroma-db-app
  ```
- Run the notebook cells to execute a workflow that creates a vector database and a keyword index. These will be available as artifacts for further use.

## Serving

- Set up the Together.ai API key in the main.py file.
- Deploy the FastAPI app:
  ```
  ENABLE_UNION_SERVING=1 REGISTRY=<YOUR_REGISTRY> union deploy apps -p demo app.py contextual-rag-fastapi-app
  ```
- Deploy the Gradio app:
  ```
  ENABLE_UNION_SERVING=1 REGISTRY=<YOUR_REGISTRY> union deploy apps -p demo app.py contextual-rag-gradio-app
  ```

## Queries

- What did Paul Graham do growing up?
- What did the author do during his time in art school?
- Can you give me a summary of the author's life?
- What did the author do during his time at Yale?
- What did the author do during his time at YC?
