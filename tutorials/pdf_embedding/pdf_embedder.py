import typing
from typing import List

import nltk
import numpy as np
from flytekit import ImageSpec, dynamic, Resources, Secret, workflow
from flytekit.core.utils import timeit
from flytekit.types.file import FlyteFile
from sentence_transformers import SentenceTransformer
import fitz
from union.actor import ActorEnvironment

embedding_image = ImageSpec(
    # requirements="requirements.txt",
    packages=[
        "datasets",
        "sentence_transformers",
        "pandas",
        "pymupdf",
        "numpy<2.0.0",
        "nltk",
        "union>=0.1.45",
        "requests>=2.29.0",
    ],
    python_version="3.11",
    registry="ghcr.io/unionai-oss",
)


cpu_actor = ActorEnvironment(
    name="pdf-actor",
    replica_count=2,
    parallelism=1,
    backlog_length=2,
    ttl_seconds=300,
    requests=Resources(cpu="2", mem="8Gi"),
    container_image=embedding_image,
)

# gpu_actor = ActorEnvironment(
#     name="embedding-actor",
#     replica_count=1,
#     parallelism=1,
#     backlog_length=5,
#     ttl_seconds=300,
#     requests=Resources(gpu="1", mem="8Gi", cpu="4"),
#     accelerator=A10G,
#     container_image=embedding_image,
#     # secrets=secret_requests=[Secret(key="hf_token")],
# )

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-en"

encoder = None
nlp_init = False


@timeit("load_nlp_model")
def load_nlp_model():
    global nlp_init
    if nlp_init:
        return
    nltk.download("punkt")
    nlp_init = True


@cpu_actor(cache=True, cache_version="1.0")
def pdf_to_text(pdf_file: FlyteFile) -> typing.List[str]:
    load_nlp_model()
    pdf_file.download()
    doc = fitz.open(filename=pdf_file.path)
    all_sentences = []
    for page_num in range(len(doc)):
        # Load the page
        page = doc.load_page(page_num)
        text = page.get_text()
        # Extract sentences
        sentences = nltk.sent_tokenize(text)
        # Add the sentences to the list
        all_sentences.extend(sentences)
    return all_sentences


@timeit("load_model")
def load_model(model_name: str = 'msmarco-MiniLM-L-6-v3') -> SentenceTransformer:
    global encoder
    if encoder:
        return encoder
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = 256
    return encoder


@cpu_actor(cache=True, cache_version="1.0")
def encode(embedding_model_name: str, text: List[str], batch_size: int = 64) -> List[List[float]]:
    model = load_model(embedding_model_name)
    embeddings: np.ndarray = model.encode(text, batch_size=batch_size)
    return [e.round(6).tolist() for e in embeddings]


@dynamic(container_image=embedding_image, requests=Resources(mem="8Gi"), cache=True, cache_version="1.0")
def process_all_sentences(
        embedding_model_name: str,
        sentences: List[str],
        chunk_size: int,
        batch_size: int = 64,
) -> List[List[List[float]]]:
    results = []
    for chunk, _ in enumerate(sentences[::chunk_size]):
        s = sentences[chunk:chunk + chunk_size]
        r = encode(embedding_model_name=embedding_model_name, text=s, batch_size=batch_size)
        results.append(r)
    return results


@workflow
def wf(
        pdf: FlyteFile = "https://huggingface.co/datasets/amitsaurav/req-doc-samples/resolve/main/amazon-dynamo-sosp2007.pdf",
        embedding_model: str = DEFAULT_MODEL,
        chunk_size: int = 10,
        batch_size: int = 64,
) -> (List[str], List[List[List[float]]]):
    sentences = pdf_to_text(pdf_file=pdf)
    results = process_all_sentences(
        embedding_model_name=embedding_model, sentences=sentences, chunk_size=chunk_size, batch_size=batch_size)
    return sentences, results


@workflow
def try_sentences(embedding_model: str = DEFAULT_MODEL, chunk_size: int = 10) -> typing.Tuple[
    typing.List[str], List[List[List[float]]]]:
    sentences = ["hello world", "this is a test", "this is a test", "this is a test", "this is a test"]
    results = process_all_sentences(embedding_model_name=embedding_model, sentences=sentences, chunk_size=chunk_size)
    return sentences, results
