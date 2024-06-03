import tarfile
from typing import Annotated

from flytekit import task
from flytekit.types.file import FileExt, FlyteFile


@task(
    cache=True,
    cache_version="2",
)
def compress_model(
    model: FlyteFile[Annotated[str, FileExt("PyTorchModule")]]
) -> FlyteFile:
    file_name = "model.tar.gz"
    tf = tarfile.open(file_name, "w:gz")
    tf.add(model.download(), arcname="sam_finetuned")
    tf.close()

    return FlyteFile(file_name)
