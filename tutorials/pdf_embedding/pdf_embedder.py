from typing import List

import fitz
import nltk
import numpy as np
from flytekit import ImageSpec, Resources, workflow
from flytekit.core.utils import timeit
from flytekit.types.file import FlyteFile
from sentence_transformers import SentenceTransformer
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
    builder="envd",
)

cpu_actor = ActorEnvironment(
    name="pdf-actor",
    replica_count=1,
    parallelism=1,
    backlog_length=2,
    ttl_seconds=300,
    requests=Resources(cpu="4", mem="8Gi"),
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


@timeit("load_model")
def load_model(model_name: str = 'msmarco-MiniLM-L-6-v3') -> SentenceTransformer:
    global encoder
    if encoder:
        return encoder
    print(f"Loading model {model_name}")
    encoder = SentenceTransformer(model_name)
    encoder.max_seq_length = 256
    # encoder.start_multi_process_pool()
    return encoder


def pdf_to_text(pdf_file: FlyteFile) -> List[str]:
    pdf_file.download()
    doc = fitz.open(filename=pdf_file.path)
    all_sentences = []
    for page_num in range(len(doc)):
        # Load the page
        page = doc.load_page(page_num)
        text = page.get_text()
        # Extract sentences
        sentences = nltk.sent_tokenize(text)
        all_sentences.extend(sentences)
    return all_sentences


@cpu_actor.task(cache=False, cache_version="1.0", enable_deck=True)
def pdf_to_embeddings(
        embedding_model_name: str,
        pdf_file: FlyteFile,
        batch_size: int = 64,
) -> np.ndarray:
    load_nlp_model()
    model = load_model(embedding_model_name)
    sentences = pdf_to_text(pdf_file)
    arr = model.encode(sentences, batch_size=batch_size, show_progress_bar=True)
    print(f"Processed {len(sentences)} sentences")
    return arr


@workflow
def embed_single_pdf(
        pdf: FlyteFile = "https://huggingface.co/datasets/amitsaurav/req-doc-samples/resolve/main/amazon-dynamo-sosp2007.pdf",
        embedding_model: str = DEFAULT_MODEL,
        batch_size: int = 64,
) -> np.ndarray:
    return pdf_to_embeddings(embedding_model_name=embedding_model, pdf_file=pdf, batch_size=batch_size)


if __name__ == '__main__':
    out_arr = embed_single_pdf()
    print(out_arr.shape)
