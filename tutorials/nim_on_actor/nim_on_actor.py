# # Serving NVIDIA NIM Models with Union Actors
#
# This tutorial shows you how to serve NVIDIA NIM-supported models using Union actors.
#
# By using Union actors, we ensure the model is pulled from the model registry
# and initialized only once. This setup guarantees the model remains available for serving
# as long as the actor is running, enabling efficient inference.
#
# Using this approach, we’re upgrading from batch inference to near-real-time inference,
# taking advantage of Union actors to improve performance.
#
# Let’s dive in by importing the necessary libraries and modules:

import os
from typing import Iterator
import functools

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
    registry=os.getenv("IMAGE_SPEC_REGISTRY"),
    packages=[
        "langchain-nvidia-ai-endpoints==0.3.5",
        "langchain==0.3.7",
        "langchain-community==0.3.7",
        "arxiv==2.1.3",
        "pymupdf==1.24.14",
        "union==0.1.117",
        "flytekitplugins-inference==1.14.3",
    ],
)

# ## Loading Arxiv data
#
# In this step, we load the Arxiv data using LangChain.
# You can adjust the `load_max_docs` parameter to a higher value to retrieve more documents from the Arxiv repository.


@fl.task(
    cache=True,
    cache_version="0.1",
    container_image=image,
)
def load_arxiv() -> Iterator[str]:
    from langchain_community.document_loaders import ArxivLoader

    loader = ArxivLoader(
        query="reasoning", load_max_docs=5, load_all_available_meta=False
    )
    documents = loader.load()

    for document in documents:
        yield document.page_content


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

# Setting the replica count to 1 in the actor to ensure the model is served once and reused for generating predictions.
# The NIM pod template is configured within the actor definition.
# The TTL (Time-To-Live) is set to 300 seconds, meaning the actor will remain active for 300 seconds without any tasks running.
# An A10G GPU is used to serve the model, ensuring optimal performance.

actor_env = ActorEnvironment(
    name="nim-actor",
    replica_count=1,
    pod_template=nim_instance.pod_template,
    container_image=image,
    ttl_seconds=300,
    secret_requests=[fl.Secret(key=HF_KEY), fl.Secret(key=NGC_KEY)],
    accelerator=A10G,
    requests=fl.Resources(gpu="0"),
)

# ## Defining an actor task
#
# In this step, we define an actor task to generate summaries of Arxiv PDFs.
# The task uses the LLama3 model in combination with LangChain for summarization.


@actor_env.task
def generate_summary(pdf: str, repo_id: str) -> str:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    os.environ["NVIDIA_API_KEY"] = fl.current_context().secrets.get(key=NGC_KEY)

    llm = ChatNVIDIA(
        base_url=f"{nim_instance.base_url}/v1", model=repo_id.split("/")[1]
    )

    prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
    chain = create_stuff_documents_chain(llm, prompt)

    return chain.invoke({"context": [Document(page_content=pdf[:8192])]})


# ## Defining a workflow
#
# Here, we set up a workflow that first loads the data and then summarizes it.
# We use a map task to generate summaries in parallel. Since the replica count is set to 1,
# only one map task runs at a time. However, if you increase the replica count, more tasks will run concurrently,
# spinning up additional models for faster serving.
#
# After the first run, subsequent summarization tasks reuse the actor environment, speeding up the process.
#
# The workflow returns a list of summaries.


@fl.workflow
def batch_inference_wf(repo_id: str = HF_REPO_ID) -> list[str]:
    arxiv_pdfs = load_arxiv()
    return fl.map_task(functools.partial(generate_summary, repo_id=repo_id))(
        pdf=arxiv_pdfs
    )
