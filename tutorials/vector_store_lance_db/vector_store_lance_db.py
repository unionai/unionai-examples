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
        "pyarrow",
        "pymupdf",
        "sentence-transformers",
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

    for delimiter in config.chunk_delimiters:
        _chunks = document.split(delimiter)
        if max(len(x) for x in _chunks) < config.approximate_chunk_size:
            break

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
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer

    documents_dir.download()
    documents_dir: Path = Path(documents_dir)

    document_chunks: list[list[str]] = []
    print("chunking documents")

    paper_ids = []
    for document_fp in documents_dir.glob("**/*.txt"):
        with open(document_fp, "rb") as f:
            document = f.read().decode("utf-8")
            chunks = chunk_document(document, config)
            document_chunks.append(chunks)
            paper_ids.append(document_fp.stem)

    model = SentenceTransformer(EMBEDDING_MODEL)
    document_vectors = []
    vectors = model.encode(document_chunks)
    print("embedding documents")
    for chunks in document_chunks:
        document_vectors.append(model.encode(chunks))

    data = []
    for paper_id, vectors, chunks in zip(paper_ids, document_vectors, document_chunks):
        for vector, chunk in zip(vectors, chunks):
            data.append(
                {
                    "paper_id": paper_id,
                    "vector": vector,
                    "text": chunk,
                }
            )

    lancedb_dir = f"{union.current_context().working_directory}/lancedb"
    db = lancedb.connect(lancedb_dir)
    schema = pa.schema(
        {
            "paper_id": pa.string(),
            "vector": pa.list_(pa.float32(), len(document_vectors[0][0])),
            "text": pa.string(),
        }
    )
    table = db.create_table("papers", data, schema=schema)
    table.create_scalar_index("paper_id", index_type="BITMAP")
    return union.FlyteDirectory(lancedb_dir), TestQuery(document_chunks[0][0], paper_ids[0])


@union.task(container_image=image)
def validate_vector_store(
    vector_store: union.FlyteDirectory,
    test_query: TestQuery,
):
    import lancedb
    from numpy.testing import assert_allclose
    from sentence_transformers import SentenceTransformer

    vector_store.download()
    db = lancedb.connect(vector_store.path)
    model = SentenceTransformer(EMBEDDING_MODEL)
    table = db.open_table("papers")
    vector_query = model.encode(test_query.query)
    result = (
        table.search(vector_query)
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
    arxiv_papers = download_arxiv_papers(query, max_results)
    parsed_papers = extract_documents(arxiv_papers)
    vector_store, test_query = create_vector_store(parsed_papers, config)
    validate_vector_store(vector_store, test_query)
    return vector_store
