# # Serving NVIDIA NIM Models with Union Actors
#
# This tutorial shows you how to serve NVIDIA NIM-supported models using Union actors.

# {{run-on-union}}

# By using Union actors, we ensure the model is pulled from the model registry
# and initialized only once. This setup guarantees the model remains available for serving
# as long as the actor is running, enabling "near-real-time" inference.
#
# Let’s dive in by importing the necessary libraries and modules:

import functools
import os

import flytekit as fl
from flytekit.extras.accelerators import A10G
from flytekitplugins.inference import NIM, NIMSecrets
from union.actor import ActorEnvironment

# ## Creating secrets
#
# This workflow requires both a Hugging Face API key and an NGC API key. Below are the steps to set up these secrets:
#
# ### Setting up the Hugging Face secret
#
# 1. **Generate an API key:** Obtain your API key from the Hugging Face website.
# 2. **Create a Secret:** Use the Union CLI to create the secret:
#
# ```bash
# union create secret hf-api-key
# ```
#
# ### Setting up the NGC secret
#
# 1. **Generate an API key:** Obtain your API key from the [NGC website](https://org.ngc.nvidia.com/setup/personal-keys).
# 2. **Create a secret:** Use the Union CLI to create the secret:
#
# ```bash
# union create secret ngc-key
# ```
#
# ### Creating the image pull secret
#
# To pull the container image from NGC, you need to create a Docker registry secret manually. Run the following command:
#
# ```bash
# kubectl create -n <PROJECT>-<DEMO> secret docker-registry nvcrio-cred \
#   --docker-server=nvcr.io \
#   --docker-username='$oauthtoken' \
#   --docker-password=<YOUR_NGC_TOKEN>
# ```
#
# ### Key details
#
# - **`NGC_IMAGE_SECRET`:** Required to pull the container image from NGC.
# - **`NGC_KEY`:** Used to pull models from NGC after the container is up and running.

HF_KEY = "hf-api-key"
HF_REPO_ID = "Samhita/OrpoLlama-3-8B-Instruct"

NGC_KEY = "ngc-key"
NGC_IMAGE_SECRET = "nvcrio-cred"

# ## Defining the imagespec
#
# We include all the necessary libraries in the imagespec to ensure they are available when executing the workflow.

image = fl.ImageSpec(
    name="nim_serve",
    builder="union",
    registry=os.getenv("IMAGE_SPEC_REGISTRY"),
    packages=[
        "langchain-nvidia-ai-endpoints==0.3.5",
        "langchain==0.3.7",
        "langchain-community==0.3.7",
        "arxiv==2.1.3",
        "pymupdf==1.25.1",
        "union==0.1.117",
        "flytekitplugins-inference==1.14.3",
    ],
)

# ## Loading Arxiv data
#
# In this step, we load the Arxiv data using LangChain.
# You can adjust the `top_k_results` parameter to a higher value to retrieve more documents from the Arxiv repository.


@fl.task(
    cache=True,
    cache_version="0.5",
    container_image=image,
)
def load_arxiv() -> list[list[str]]:
    from langchain_community.document_loaders import ArxivLoader

    loader = ArxivLoader(query="reasoning", top_k_results=100, doc_content_chars_max=8000)

    documents = []
    temp_documents = []
    for document in loader.lazy_load():
        temp_documents.append(document.page_content)

        if len(temp_documents) >= 10:
            documents.append(temp_documents)
            temp_documents = []

    return documents


# We return documents in batches of 10 and send them to the model to generate summaries in parallel.
#
# ## Instanting NIM and defining an actor environment
#
# We instantiate the NIM plugin and set up the actor environment.
# We load a fine-tuned LLama3 8B model to serve.

nim_instance = NIM(
    image="nvcr.io/nim/meta/llama3-8b-instruct:1.0.0",
    secrets=NIMSecrets(
        ngc_image_secret=NGC_IMAGE_SECRET,
        ngc_secret_key=NGC_KEY,
        secrets_prefix="_UNION_",
        hf_token_key=HF_KEY,
    ),
    hf_repo_ids=[HF_REPO_ID],
    lora_adapter_mem="500Mi",
    env={"NIM_PEFT_SOURCE": "/home/nvs/loras"},
)

# By default, the NIM instantiation sets cpu, gpu, and mem to 1, 1, and 20Gi, respectively. You can modify these settings as needed.
#
# Setting the replica count to 1 in the actor to ensure the model is served once and reused for generating predictions.
# The NIM pod template is configured within the actor definition.
# The TTL (Time-To-Live) is set to 900 seconds, meaning the actor will remain active for 900 seconds without any tasks running.
# An A10G GPU is used to serve the model, ensuring optimal performance.
# The `gpu` parameter is set to `0` to allocate the GPU for the model server, rather than for the Flyte task.

actor_env = ActorEnvironment(
    name="nim-actor",
    replica_count=1,
    pod_template=nim_instance.pod_template,
    container_image=image,
    ttl_seconds=900,
    secret_requests=[fl.Secret(key=HF_KEY), fl.Secret(key=NGC_KEY)],
    accelerator=A10G,
    requests=fl.Resources(gpu="0"),
)

# ## Defining an actor task
#
# In this step, we define an actor task to generate summaries of Arxiv PDFs.
# The task uses the LLama3 model in combination with LangChain for summarization.


@actor_env.task
def generate_summary(arxiv_pdfs: list[str], repo_id: str) -> list[str]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain.prompts import PromptTemplate

    os.environ["NVIDIA_API_KEY"] = fl.current_context().secrets.get(key=NGC_KEY)

    llm = ChatNVIDIA(base_url=f"{nim_instance.base_url}/v1", model=repo_id.split("/")[1])

    prompt_template = "Summarize this content: {content}"
    prompt = PromptTemplate(input_variables=["content"], template=prompt_template)

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain.batch([{"content": arxiv_pdf} for arxiv_pdf in arxiv_pdfs])


# A batch of PDFs is provided as input to the task, allowing the PDFs to be processed
# simultaneously to generate summaries. When the task is invoked multiple times,
# subsequent batches will reuse the actor environment.
#
# ## Defining a workflow
#
# Here, we set up a workflow that first loads the data and then summarizes it.
# We use a map task to generate summaries in parallel. Since the replica count is set to 1,
# only one map task runs at a time. However, if you increase the replica count, more tasks will run concurrently,
# spinning up additional models.
#
# After the first run, subsequent summarization tasks reuse the actor environment, speeding up the process.
#
# The workflow returns a list of summaries.


@fl.workflow
def batch_inference_wf(repo_id: str = HF_REPO_ID) -> list[list[str]]:
    arxiv_pdfs = load_arxiv()
    return fl.map_task(functools.partial(generate_summary, repo_id=repo_id))(
        arxiv_pdfs=arxiv_pdfs
    )
