import re
from pathlib import Path

from Bio import Entrez

from ollama.constants import ENTREZ_EMAIL

Entrez.email = ENTREZ_EMAIL


EXAMPLE_TEMPLATE = """<|system|>You are a medical research assistant AI that has
been fine-tuned on the latest research. Use the latest knowledge beyond your
initial training data cutoff to provide the most up-to-date information.<|end|>

<|user|>What's the latest medical research relating to {query}?<|end|>

<|assistant|>
{abstracts}
<|end|>
"""


def get_abstracts(query: str, top_n: int) -> list[str]:
    search_handle = Entrez.esearch(
        db="pubmed",
        retmax=top_n,
        term=query,
        idtype="acc",
    )
    search_results = Entrez.read(search_handle)

    abstracts = []
    for _id in search_results["IdList"]:
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=_id,
            rettype="abstract",
            retmode="text",
        )
        abstract = fetch_handle.read()
        abstract = re.sub("\d+\. ", "", abstract, count=1)
        abstracts.append(abstract)
    return abstracts


def create_dataset(
    output_dir: Path,
    queries: list[str],
    top_n: int = 3,
):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for query in queries:
        abstracts = get_abstracts(query, top_n)
        output_path = output_dir / f"{query.replace(' ', '_')}.txt"
        print(f"writing example to: {output_path}")
        example = EXAMPLE_TEMPLATE.format(
            query=query,
            abstracts=f"\n\n{'-' * 20}\n\n".join(abstracts),
        )
        with output_path.open("w") as f:
            f.write(example)

    print(f"created dataset at: {output_dir}")
