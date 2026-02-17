# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "sentence-transformers>=2.2.0",
#    "chromadb>=0.4.0",
#    "requests>=2.31.0",
# ]
# main = "embedding_pipeline"
# ///

"""
Quote Embedding Pipeline

Fetches quotes from a public API and creates embeddings using sentence-transformers.
The embeddings are stored in a ChromaDB database that can be used for semantic search.
"""

# {{docs-fragment imports}}
import os
import tempfile

import flyte
from flyte.io import Dir
# {{/docs-fragment imports}}

# {{docs-fragment embedding-env}}
# Define the embedding environment
embedding_env = flyte.TaskEnvironment(
    name="quote-embedding",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "requests>=2.31.0",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    cache="auto",
)
# {{/docs-fragment embedding-env}}


# {{docs-fragment fetch-quotes}}
@embedding_env.task
async def fetch_quotes() -> list[dict]:
    """
    Fetch quotes from a public quotes API.

    Returns:
        List of quote dictionaries with 'quote' and 'author' fields.
    """
    import requests

    print("Fetching quotes from API...")
    response = requests.get("https://dummyjson.com/quotes?limit=100")
    response.raise_for_status()

    data = response.json()
    quotes = data.get("quotes", [])

    print(f"Fetched {len(quotes)} quotes")
    return quotes
# {{/docs-fragment fetch-quotes}}


# {{docs-fragment embed-quotes}}
@embedding_env.task
async def embed_quotes(quotes: list[dict]) -> Dir:
    """
    Create embeddings for quotes and store them in ChromaDB.

    Args:
        quotes: List of quote dictionaries with 'quote' and 'author' fields.

    Returns:
        Directory containing the ChromaDB database.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create ChromaDB in a temporary directory
    db_dir = tempfile.mkdtemp()
    print(f"Creating ChromaDB at {db_dir}...")

    client = chromadb.PersistentClient(path=db_dir)
    collection = client.create_collection(
        name="quotes",
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data for insertion
    texts = [q["quote"] for q in quotes]
    ids = [str(q["id"]) for q in quotes]
    metadatas = [{"author": q["author"], "quote": q["quote"]} for q in quotes]

    print(f"Embedding {len(texts)} quotes...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=texts,
    )

    print(f"Stored {len(quotes)} quotes in ChromaDB")
    return await Dir.from_local(db_dir)
# {{/docs-fragment embed-quotes}}


# {{docs-fragment embedding-pipeline}}
@embedding_env.task
async def embedding_pipeline() -> Dir:
    """
    Main pipeline that fetches quotes and creates embeddings.

    Returns:
        Directory containing the ChromaDB database with quote embeddings.
    """
    print("Starting embedding pipeline...")

    # Fetch quotes from API
    quotes = await fetch_quotes()

    # Create embeddings and store in ChromaDB
    db_dir = await embed_quotes(quotes)

    print("Embedding pipeline complete!")
    return db_dir
# {{/docs-fragment embedding-pipeline}}


# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(embedding_pipeline)
    print(f"Embedding run URL: {run.url}")
    run.wait()
    print(f"Embedding complete! Database directory: {run.outputs()}")
# {{/docs-fragment main}}
