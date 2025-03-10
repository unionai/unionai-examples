import os
from subprocess import run
from typing import Annotated

import union
from flytekit.extras.accelerators import A100, L4
from flytekit.types.directory import FlyteDirectory
from flytekit.image_spec.image_spec import ImageBuildEngine


EMBEDDING_MODEL = "snowflake/arctic-embed-l:1.0.1"
RERANKER_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3:1.0.2"
LLM_MODEL = "meta/llama-3.1-8b-instruct:1.3.3"

NIMEmbeddingModel = union.Artifact(name="nim-snowflake-arctic-embed-l")
NIMRerankerModel = union.Artifact(name="nim-nv-rerankqa-mistral-4b-v3")
NIMLLMModel = union.Artifact(name="llama-31-8b-instruct")

enterprise_rag_embedding_image = union.ImageSpec(
    name="enterprise-rag-embedding",
    base_image=union.ImageSpec(
        name="enterprise-rag-embedding",
        base_image=f"nvcr.io/nim/{EMBEDDING_MODEL}",
        builder="default",
        registry=os.getenv("REGISTRY"),
    ),
    packages=["union==0.1.151"],
)

enterprise_rag_reranker_image = union.ImageSpec(
    name="enterprise-rag-reranker",
    base_image=union.ImageSpec(
        name="enterprise-rag-reranker",
        base_image=f"nvcr.io/nim/{RERANKER_MODEL}",
        builder="default",
        registry=os.getenv("REGISTRY"),
    ),
    packages=["union==0.1.151"],
)
ImageBuildEngine.build(enterprise_rag_reranker_image)

enterprise_rag_llm_image = union.ImageSpec(
    name="enterprise-rag-llm",
    base_image=union.ImageSpec(
        name="enterprise-rag-llm",
        base_image=f"nvcr.io/nim/{LLM_MODEL}",
        builder="default",
        registry=os.getenv("REGISTRY"),
    ),
    packages=["union==0.1.151"],
)
ImageBuildEngine.build(enterprise_rag_llm_image)


@union.task(
    container_image=enterprise_rag_embedding_image,
    requests=union.Resources(cpu="3", mem="16Gi", ephemeral_storage="16Gi", gpu="1"),
    secret_requests=[union.Secret(key="nvidia-api-key", env_var="NGC_API_KEY")],
    accelerator=L4,
)
def create_model() -> FlyteDirectory:
    ctx = union.current_context()
    working_dir = ctx.working_directory
    nim_cache = os.path.join(working_dir, "nim_cache")

    os.environ["LOCAL_NIM_CACHE"] = nim_cache
    os.environ["NIM_CACHE_PATH"] = nim_cache
    os.environ["NGC_HOME"] = os.path.join(nim_cache, "ngc", "hub")

    run(["download-to-cache"])
    return nim_cache


@union.workflow
def download_nim_models_to_cache() -> tuple[
    Annotated[FlyteDirectory, NIMEmbeddingModel],
    Annotated[FlyteDirectory, NIMRerankerModel],
    Annotated[FlyteDirectory, NIMLLMModel],
]:
    return (
        create_model(),
        create_model().with_overrides(
            container_image=enterprise_rag_reranker_image.image_name(),
            requests=union.Resources(
                cpu="2", mem="16Gi", ephemeral_storage="16Gi", gpu="1"
            ),
            accelerator=L4,
        ),
        create_model().with_overrides(
            container_image=enterprise_rag_llm_image.image_name(),
            requests=union.Resources(
                cpu="2", mem="20Gi", ephemeral_storage="16Gi", gpu="1"
            ),
            accelerator=A100,
        ),
    )
