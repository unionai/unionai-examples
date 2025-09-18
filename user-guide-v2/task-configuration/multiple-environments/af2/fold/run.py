import flyte
from flyte.io import File

FOLD_PACKAGES = ["ruff"]

fold_image = flyte.Image.from_debian_base().with_pip_packages(*FOLD_PACKAGES)

fold_env = flyte.TaskEnvironment(name="fold_env", image=fold_image)


@fold_env.task
def run_fold(sequence: str, msa: File) -> list[str]:
    with msa.open_sync("r") as f:
        msa_content = f.read()
    return [msa_content, sequence]
