# # Serve NVIDIA NIM Models with Union Actors
#
# This tutorial shows you how to serve NVIDIA NIM-supported models using Union actors.

# {{run-on-union}}

# By using Union actors, we ensure the model is pulled from the model registry
# and initialized only once. This setup guarantees the model remains available for serving
# as long as the actor is running, enabling "near-real-time" inference.
#
# Let's dive in by importing the necessary libraries and modules:

import functools
import os

import union
from flytekit.extras.accelerators import A10G
from flytekitplugins.inference import NIM, NIMSecrets
from union.actor import ActorEnvironment

# ## Create secrets
#
# This workflow requires both a Hugging Face API key and an NGC API key. Below are the steps to set up these secrets:

# ### Hugging Face secret
#
# 1. **Generate an API key:** Obtain your API key from the Hugging Face website.
# 2. **Create a Secret:** Use the Union CLI to create the secret:
#
# ```shell
# $ union create secret hf-api-key
# ```

# ### NGC secret
#
# 1. **Generate an API key:** Obtain your API key from the [NGC website](https://org.ngc.nvidia.com/setup).
# 2. **Create a secret:** Use the Union CLI to create the secret:
#
# ```shell
# $ union create secret ngc-api-key
# ```

# ### Image pull secret
#
# Union's remote image builder enables pulling images from private registries.
# To create an image pull secret, follow these steps:
#
# 1. Log in to NVCR locally by running: `docker login nvcr.io`
# 2. Create an image pull secret using your local `~/.docker/config.json`: `IMAGEPULLSECRET=$(union create imagepullsecret --registries nvcr.io -q)`
# 3. Add it as a Union secret: `union create secret --type image-pull-secret --value-file $IMAGEPULLSECRET nvcr-pull-creds`

HF_KEY = "hf-api-key"
HF_REPO_ID = "Samhita/OrpoLlama-3-8B-Instruct"

NGC_KEY = "ngc-api-key"
NVCR_SECRET = "nvcr-pull-creds"

# ## Define `ImageSpec`
#
# We include all the necessary libraries in the imagespec to ensure they are available when running the workflow.

image = union.ImageSpec(
    name="nim_serve",
    builder="union",
    packages=[
        "langchain-nvidia-ai-endpoints==0.3.10",
        "langchain==0.3.25",
        "langchain-community==0.3.25",
        "arxiv==2.2.0",
        "pymupdf==1.26.1",
        "union==0.1.183",
        "flytekitplugins-inference",
    ],
    builder_options={"imagepull_secret_name": NVCR_SECRET},
)

# `builder_options` is used to pass the image pull secret to the builder.

# ## Load documents from Arxiv
#
# In this step, we load the Arxiv data using LangChain.
# You can adjust the `top_k_results` parameter to a higher value to retrieve more documents from the Arxiv repository.


@union.task(cache=True, container_image=image)
def load_arxiv() -> list[list[str]]:
    from langchain_community.document_loaders import ArxivLoader

    loader = ArxivLoader(
        query="reasoning", top_k_results=10, doc_content_chars_max=8000
    )

    documents = []
    temp_documents = []
    for document in loader.lazy_load():
        temp_documents.append(document.page_content)

        if len(temp_documents) >= 10:
            documents.append(temp_documents)
            temp_documents = []

    return documents


# We return documents in batches of 10 and send them to the model to generate summaries in parallel.

# ## Set up NIM and actor
#
# We instantiate the NIM plugin and set up the actor environment.
# We load a fine-tuned LLama3 8B model to serve.

nim_instance = NIM(
    image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
    secrets=NIMSecrets(
        ngc_secret_key=NGC_KEY,
        secrets_prefix="_UNION_",
        hf_token_key=HF_KEY,
    ),
    hf_repo_ids=[HF_REPO_ID],
    lora_adapter_mem="500Mi",
    env={"NIM_PEFT_SOURCE": "/home/nvs/loras"},
)

# By default, the NIM instantiation sets `cpu`, `gpu`, and `mem` to 1, 1, and 20Gi, respectively. You can modify these settings as needed.

# To serve the NIM model efficiently, we configure the actor to launch a single replica, ensuring the model is loaded once and reused across predictions.
# We set a TTL (Time-To-Live) of 900 seconds, allowing the actor to remain active for 15 minutes while idle.
# This helps reduce cold starts and enables faster response to follow-up requests.

# The model runs on an A10G GPU, and the number of GPUs is set to 0 so that the GPU is allocated to the model server itself rather than the task that invokes it.

actor_env = ActorEnvironment(
    name="nim-actor",
    replica_count=1,
    pod_template=nim_instance.pod_template,
    container_image=image,
    ttl_seconds=900,
    secret_requests=[
        union.Secret(key=HF_KEY),
        union.Secret(key=NVCR_SECRET),
        union.Secret(key=NGC_KEY),
    ],
    accelerator=A10G,
    requests=union.Resources(gpu="0"),
)

# ## Generate summaries
#
# We define an actor task to generate summaries of Arxiv PDFs.

# The task processes a batch of PDFs simultaneously to generate summaries.
# When invoked multiple times, it reuses the existing actor environment for subsequent batches.


@actor_env.task
def generate_summary(arxiv_pdfs: list[str], repo_id: str) -> list[str]:
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    os.environ["NVIDIA_API_KEY"] = union.current_context().secrets.get(key=NGC_KEY)

    llm = ChatNVIDIA(
        base_url=f"{nim_instance.base_url}/v1", model=repo_id.split("/")[1]
    )

    prompt_template = "Summarize this content: {content}"
    prompt = PromptTemplate(input_variables=["content"], template=prompt_template)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.batch([{"content": arxiv_pdf} for arxiv_pdf in arxiv_pdfs])


# Next, we set up a workflow that first loads the data and then summarizes it.
# We use a map task to generate summaries in parallel. Since the replica count is set to 1,
# only one map task runs at a time. However, if you increase the replica count, more tasks will run concurrently,
# spinning up additional models.
#
# After the first run, subsequent summarization tasks reuse the actor environment, speeding up the process.
#
# The workflow returns a list of summaries.


@union.workflow
def batch_inference_wf(repo_id: str = HF_REPO_ID) -> list[list[str]]:
    arxiv_pdfs = load_arxiv()
    return union.map_task(functools.partial(generate_summary, repo_id=repo_id))(
        arxiv_pdfs=arxiv_pdfs
    )
