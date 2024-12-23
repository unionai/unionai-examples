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

The ingestion workflow will run daily to update the vector database and the keyword index.
The vector database will be available in the app, and support for mounting and persisting the database will be added soon!

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

## Execution

Execute the notebook cells to run a workflow that generates a vector database and a keyword index. These outputs will be stored as artifacts for future use.

You can also deploy apps directly from the Jupyter notebook using the UnionRemote Python API.

Make sure to add your Together.ai API key and the registry name to the `.env` file before proceeding with deploying the apps.

## Queries

- What did Paul Graham do growing up?
- What did the author do during his time in art school?
- Can you give me a summary of the author's life?
- What did the author do during his time at Yale?
- What did the author do during his time at YC?
