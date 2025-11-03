# # Creating a RAG App with LanceDB and Google Gemini
#
# In this tutorial, we'll create a vector DB of Arxiv papers using
# [LanceDB](https://lancedb.github.io/lancedb/). Then, we'll create a simple RAG
# serving app that uses Google Gemini 2.0 to answer questions about the papers
# in the vector DB.

# {{run-on-union}}

# ## Overview
#
# This workflow downloads a set of papers from Arxiv, extracts the text from
# the papers, and creates a vector store from the text. This vector store is
# then consumed by a simple RAG serving app using FastAPI and Google Gemini Flash 2.0.
#
# First, let's import the workflow dependencies:

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import union

image = union.ImageSpec(
    packages=[
        "arxiv",
        "lancedb",
        "numpy<2",
        "pandas",
        "pyarrow",
        "pymupdf",
        "sentence-transformers",
        "tqdm",
    ],
    builder="union",
)

# Then, we define the embedding model, an `Artifact` to save the vector store,
# and a configuration for the vector store. In this case, we'll implement a
# simple chunking strategy that splits the document into chunks of approximately
# 1000 characters each.

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"

VectorStore = union.Artifact(name="ArxivPaperVectorStore")


@dataclass
class VectorStoreConfig:
    # Delimiters to split a document into chunks. The delimiters are used to
    # iteratively split the document into chunks. It will first try to split
    # the documents by the first delimiter. If the chunk is still larger than
    # the maximum chunk size, it will try to split the document by the second
    # delimiter, and so on.
    chunk_delimiters: list[str] = field(default_factory=lambda: ["\n\n", ".\n"])

    # Approximate chunk size in characters
    approximate_chunk_size: int = 1000


# Next, we define a `TestQuery` class to validate the vector store. This
# will be used to validate the vector store after it is created at the very
# end of the workflow.


@dataclass
class TestQuery:
    query: str
    paper_id: str


# ## Define the tasks
#
# Now, we define the tasks for downloading the papers, extracting the text,
# and creating the vector store.
#
# ### Download the papers
#
# First we'll use the [`arxiv`](https://github.com/lukasschwab/arxiv.py)
# python package to download the papers from Arxiv.


@union.task(container_image=image, cache=True, cache_version="0")
def download_arxiv_papers(query: str, max_results: int) -> union.FlyteDirectory:
    """Download the papers from Arxiv given a query."""
    import arxiv

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = client.results(search)
    arxiv_dir = f"{union.current_context().working_directory}/arxiv_papers"
    os.makedirs(arxiv_dir, exist_ok=True)
    print(f"downloading to {arxiv_dir}")
    for paper in results:
        print(paper.title)
        paper_id = paper.entry_id.split("/")[-1]
        paper.download_pdf(dirpath=arxiv_dir, filename=f"{paper_id}.pdf")

    return union.FlyteDirectory(arxiv_dir)


# ### Extract the text from the PDF files
#
# Then, we'll use the [`pymupdf`](https://pymupdf.readthedocs.io/en/latest/)
# python package to extract the text from the PDF files.


@union.task(container_image=image, cache=True, cache_version="0")
def extract_documents(arxiv_dir: union.FlyteDirectory) -> union.FlyteDirectory:
    """Extract raw text from the PDF files."""
    import os
    import pymupdf

    arxiv_dir.download()
    arxiv_dir: Path = Path(arxiv_dir)
    documents_dir = f"{union.current_context().working_directory}/documents"
    os.makedirs(documents_dir, exist_ok=True)
    for pdf_file in arxiv_dir.glob("**/*.pdf"):
        text_fp = f"{documents_dir}/{pdf_file.stem}.txt"
        print(f"writing text to {text_fp}")
        with open(text_fp, "wb") as f:
            for page in pymupdf.open(pdf_file):
                f.write(page.get_text().encode("utf-8"))

    return union.FlyteDirectory(documents_dir)


# ### Create the vector store
#
# Next, we define a helper function to chunk a document into smaller chunks.
# This is a simple strategy that splits the document by the delimiters and
# then consolidates the chunks such that the largest chunk is approximately
# the size specified by the `approximate_chunk_size`.


def chunk_document(document: str, config: VectorStoreConfig) -> list[str]:
    """Helper function to chunk a document into smaller chunks."""

    # Try to split the documents by iterating through the provided chunk
    # delimiters. If the largest chunk is smaller than the approximate_chunk_size,
    # we select that delimiter.
    for delimiter in config.chunk_delimiters:
        _chunks = document.split(delimiter)
        if max(len(x) for x in _chunks) < config.approximate_chunk_size:
            break

    # Consolidate the chunks such that chunks are about the size specified by
    # the approximate_chunk_size.
    chunks: list[str] = []
    _new_chunk = ""
    for chunk in _chunks:
        if len(_new_chunk) > config.approximate_chunk_size:
            chunks.append(_new_chunk)
            _new_chunk = ""
        else:
            _new_chunk += chunk

    return chunks


# This helper function is called by the `create_vector_store` task below,
# which chunks the documents, embeds the chunks using the [`SentenceTransformer`](https://sbert.net/)
# library, and then adds the chunks to the vector store.


@union.task(
    container_image=image,
    requests=union.Resources(
        cpu="4",
        mem="8Gi",
        ephemeral_storage="8Gi",
    ),
)
def create_vector_store(
    documents_dir: union.FlyteDirectory,
    config: VectorStoreConfig,
) -> tuple[Annotated[union.FlyteDirectory, VectorStore], TestQuery]:
    """Create a vector store from a directory of documents."""
    import lancedb
    import pyarrow as pa
    import tqdm

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create a LanceDB database and table.
    lancedb_dir = f"{union.current_context().working_directory}/lancedb"
    db = lancedb.connect(lancedb_dir)
    papers_table = db.create_table(
        "papers",
        schema=pa.schema(
            {
                "paper_id": pa.string(),
                "vector": pa.list_(pa.float32(), model.get_sentence_embedding_dimension()),
                "text": pa.string(),
            }
        ),
    )
    paper_ids_table = db.create_table(
        "paper_ids",
        schema=pa.schema(
            {
                "paper_id": pa.string(),
                "text": pa.string(),
            }
        ),
    )

    documents_dir.download()
    documents_dir: Path = Path(documents_dir)

    document_paths = list(documents_dir.glob("**/*.txt"))

    # Iterate through the documents and add them to the vector store.
    for i, document_fp in tqdm.tqdm(
        enumerate(document_paths),
        total=len(document_paths),
        desc="chunking and embedding documents",
    ):

        with open(document_fp, "rb") as f:
            document = f.read().decode("utf-8")

        chunks = chunk_document(document, config)
        vectors = model.encode(chunks)
        data = [
            {
                "paper_id": document_fp.stem,
                "vector": vector,
                "text": chunk,
            }
            for vector, chunk in zip(vectors, chunks)
        ]
        papers_table.add(data)
        # only add the first few characters of the first chunk to the paper_ids_table
        paper_ids_table.add([{"paper_id": document_fp.stem, "text": chunks[0][:200]}])
        if i == 0:
            test_query = TestQuery(data[0]["text"], data[0]["paper_id"])

    papers_table.create_scalar_index("paper_id", index_type="BITMAP")
    paper_ids_table.create_scalar_index("paper_id", index_type="BITMAP")
    return union.FlyteDirectory(lancedb_dir), test_query


# ### Validate the vector store
#
# The last step of our workflow is to define a validation task that queries the
# vector store using the `TestQuery` that we produced in the previous step.
# This is a simple strategy to make sure that the vector store is working as
# intended.


@union.task(container_image=image)
def validate_vector_store(
    vector_store: union.FlyteDirectory,
    test_query: TestQuery,
):
    """Validate the vector store by querying it with a test query.

    Test query should be a document from the vector store, which guarantees
    that the vector store returns a result with close-to-zero distance.
    """
    import lancedb
    from numpy.testing import assert_allclose
    from sentence_transformers import SentenceTransformer

    vector_store.download()
    db = lancedb.connect(vector_store.path)
    model = SentenceTransformer(EMBEDDING_MODEL)
    table = db.open_table("papers")
    vector_query = model.encode(test_query.query)
    result = (
        table.search(vector_query, "vector")
        .limit(1)
        .where(f"paper_id = '{test_query.paper_id}'")
        .to_list()
    )
    assert len(result) == 1
    assert all(isinstance(x, float) for x in result[0]["vector"])
    assert result[0]["text"] == test_query.query
    assert result[0]["paper_id"] == test_query.paper_id
    assert_allclose(result[0]["_distance"], 0.0, atol=0.0001)
    print("✅ test query passed")


# ## Define the workflow
#
# Now, we define the workflow that puts all of the tasks together.


@union.workflow
def main(
    query: str,
    max_results: int,
    config: VectorStoreConfig = VectorStoreConfig(),
) -> union.FlyteDirectory:
    """Main workflow to create a vector store from Arxiv papers."""
    arxiv_papers = download_arxiv_papers(query, max_results)
    parsed_papers = extract_documents(arxiv_papers)
    vector_store, test_query = create_vector_store(parsed_papers, config)
    validate_vector_store(vector_store, test_query)
    return vector_store


# You can run the workflow using the following command:
#
# ```shell
# $ union run --remote vector_store_lance_db.py main --query "artificial intelligence" --max_results 10
# ```

# ## Deploying a RAG FastAPI App
#
# In this section, we'll deploy a FastAPI Retrieval Augmented Generation (RAG) app
# that uses Google Gemini 2.0 Flash and the vector store we created above to
# answer questions about the papers.
#
# First, create a `google_api_key`: https://ai.google.dev/gemini-api/docs/api-key
#
# Then, create a `google_api_key` secret in Union:
#
# ```shell
# $ union create secret --name google_api_key --value <your_google_api_key>
# ```
#
# Below we'll define the Union `App` configuration:

from union.app import App, Input

fastapi_image = union.ImageSpec(
    name="arxiv-rag-base-image",
    builder="union",
    packages=[
        "lancedb",
        "fastapi[standard]",
        "google-genai",
        "pyarrow",
        "sentence-transformers",
        "union-runtime",
    ],
)

fastapi_app = App(
    name="arxiv-rag-fastapi-app",
    include=["fastapi_app.py"],
    args="fastapi dev fastapi_app.py --port 8082",
    port=8082,
    container_image=fastapi_image,
    inputs=[
        Input(
            value=VectorStore.query(),
            download=True,
            env_var="VECTOR_STORE_PATH",
        ),
    ],
    secrets=[union.Secret(key="google_api_key", env_var="GOOGLE_API_KEY")],
    limits=union.Resources(cpu="1", mem="2Gi", ephemeral_storage="4Gi"),
    requires_auth=False,
)

# In the code above, you can see the following:
# - The FastAPI app code is included in the `fastapi_app.py` file, which we specify in the `include` argument.
#   The `fastapi_app.py` file can be found [here](https://github.com/unionai/unionai-examples/blob/main/v1/tutorials/vector_store_lance_db/fastapi_app.py).
# - The `args` argument specifies the command to run the app with. In this case, we're using `fastapi dev --port 8082` to run the app in development mode on port 8082.
# - The app configuration uses the `fastapi_image` as the container image that the app runs on.
# - We bind the `VectorStore` artifact as an input to the app, downloading the vector store
#   to the app's filesystem on startup, and bind it to the `VECTOR_STORE_PATH` environment variable.
# - We request a `google_api_key` secret from Union, which is used to authenticate requests to the Gemini API.
# - We request 1 CPU, 2GB of memory, and 4GB of ephemeral storage for the app.
# - We set `requires_auth=False` to allow unauthenticated access to the app. This value is `True` by default.
#   To implement authentication, see [this example](https://github.com/unionai/unionai-examples/blob/main/v1/tutorials/serving_webhook/main.py)
#
# ### Deploying the app
#
# Then, you can deploy the app using the following command:
#
# ```shell
# $ union deploy apps vector_store_lance_db.py arxiv-rag-fastapi-app
# ```
#
# This will produce an `{endpoint}` URL that you can use to call the app, which
# will look something like `https://gifted-elephant-xyz.apps.demo.union.ai`.
#
# ### Calling the app
#
# Use `curl` to call the app's endpoints:
#
# ```shell
# $ export ENDPOINT="<ADD_ENDPOINT_HERE>""
# ```
#
# To get the available papers, you can call the `/papers` endpoint:
#
# ```shell
# $ curl --no-buffer "$ENDPOINT/papers"
# ```
#
# You'll see an output that looks like:
#
# ```
# [{"paper_id":"2504.01911v1","text":"Advancing AI-Scientist Understanding ...}]
# ```
#
# Use the paper IDs to ask a question about a specific paper:
#
# ```shell
# $ export PAPER_ID="<ADD_PAPER_ID_HERE>"
# ```
#
# To ask a question about a specific paper, you can call the `/ask_paper/{paper_id}` endpoint:
#
# ```shell
# $ curl --no-buffer "$ENDPOINT/ask_paper/$PAPER_ID?query=what%20is%20the%20key%20point%20of%20this%20article"
# ```
#
# To ask a question across all of the papers, you can call the `/ask` endpoint:
#
# ```shell
# $ curl --no-buffer "$ENDPOINT/ask?query=what%20is%20the%20latest%20AI%20research?"
# ```
