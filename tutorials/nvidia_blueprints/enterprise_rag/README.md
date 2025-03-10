# Enterprise RAG Blueprint

NVIDIA Blueprints are powerful Compound AI systems, but getting them into production usually requires a dedicated infrastructure/platform expert and multiple subject matter experts (SMEs) to deploy the full solution.

Union makes this process easier by turning blueprints into enterprise-ready workflows that you can develop locally and deploy to production. It provides all the necessary components to build production-grade Compound AI systems, from data processing to model serving.

## What Union Unlocks

When you convert blueprints to Union workflows, you get:

1. **Data Ingestion as a Background Job**

   Data ingestion is handled as a Union task, with built-in capabilities like:

   - **Retries**: If the vector database is down for some reason, the task automatically retries instead of failing immediately.
   - **Caching**: The task won’t rerun for the same file, so there’s no need to manually delete documents from the vector database as suggested in the blueprint. It just returns the cached output.
   - **Secrets Management**: Easily assign secrets with Union’s built-in secret interface.
   - **Simplified Image Management**: No need to write complex Dockerfiles; just use Union’s imagespec to define images.

   We're using a hosted Milvus vector database to ensure scalability and a production-grade setup.

2. **Model Serving**

   All NVIDIA NIM models can be served using Union. You can:

   - Store models as Union artifacts (like caching) and serve them later.
   - Set resource limits (GPU, CPU, memory), environment variables, secrets, and more.

3. **RAG Application Serving**

   - Beyond just models, you can also serve full RAG applications with Union, so everything is managed under one framework.
   - The RAG app tracks the progress of the background data ingestion job using UnionRemote and reports its status.
   - Unlike the blueprint, which used nested invocations, our proposed pipeline is more straightforward—it consists of a data ingestion task, FastAPI services, and a Gradio app, with no boilerplate code. Union keeps the code centralized, making development and maintenance more efficient.

4. **Development to Production**
   - Test the data ingestion task locally before deploying it.
   - Run the FastAPI app locally before deploying it as a full application on Union.
   - Moving from local development to production is straightforward.

## Why Union?

Union makes NVIDIA Blueprints enterprise-ready. No need to worry about versioning, reproducibility, caching, or deployment complexity—Union provides all the production-grade features out of the box.

Some other key benefits of using Union are [outlined in the PDF to podcast blueprint example](https://docs.union.ai/serverless/tutorials/language-models/pdf-to-podcast-blueprint#key-benefits-of-using-union).

## Execution

1. Create an account on [Union Serverless](https://signup.union.ai/).
2. Install the [Union CLI](https://docs.union.ai/serverless/user-guide/getting-started/local-setup).\
3. Create secrets:
   1. Union API key to execute the background job: `union create secret union-api-key`
   2. Milvus URI: `union create milvus-uri`
   3. Milvus token: `union creae milvus-token`
4. Register the data ingestion workflow by running: `union register ingestion.py`
5. Export your NVIDIA/NGC API key to download NIM model images from NGC catalog: `export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>`
6. Generate artifacts for the models by running: `REGISTRY=<YOUR_REGISTRY> union run --remote artifact.py download_nim_models_to_cache`
7. Deploy the models with the following commands:

   ```
    REGISTRY=<YOUR_REGISTRY> union deploy apps app.py enterprise-rag-embedding
    REGISTRY=<YOUR_REGISTRY> union deploy apps app.py enterprise-rag-reranker
    REGISTRY=<YOUR_REGISTRY> union deploy apps app.py enterprise-rag-llm
   ```

8. Deploy the RAG app by running: `union deploy apps app.py enterprise-rag`
