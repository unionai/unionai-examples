import os
from subprocess import run
from typing import Annotated

import union
from flytekit.extras.accelerators import A100, L4
from flytekit.types.directory import FlyteDirectory

from .frontend.utils import (
    NIMEmbeddingModel,
    NIMLLMModel,
    NIMRerankerModel,
    enterprise_rag_embedding_image,
    enterprise_rag_llm_image,
    enterprise_rag_reranker_image,
)


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
            container_image=enterprise_rag_reranker_image,
            requests=union.Resources(cpu="2", mem="16Gi", ephemeral_storage="16Gi"),
            accelerator=None,
        ),
        create_model().with_overrides(
            container_image=enterprise_rag_llm_image,
            requests=union.Resources(
                cpu="2", mem="20Gi", ephemeral_storage="16Gi", gpu="1"
            ),
            accelerator=A100,
        ),
    )
