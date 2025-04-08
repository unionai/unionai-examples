# # Creating a Vector Store with LanceDB
#
# In this tutorial, we'll create a vector DB of Arxiv papers using LanceDB.
# Then, we'll create a simple RAG serving app that uses Google Gemini 2.0 to
# answer questions about the papers in the vector DB.

# {{run-on-union}}

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
    ]
)

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


@dataclass
class TestQuery:
    query: str
    paper_id: str


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
