# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "main"
# params = "sequence=AAGGTTCCAA"
# ///

# {{docs-fragment import}}
import logging
import pathlib

from fold.run import fold_env, fold_image, run_fold
from msa.run import msa_env, MSA_PACKAGES, run_msa

import flyte
# {{/docs-fragment import}}

# {{docs-fragment image_and_env}}
main_image = fold_image.with_pip_packages(*MSA_PACKAGES)

env = flyte.TaskEnvironment(
    name="multi_env",
    depends_on=[fold_env, msa_env],
    image=main_image,
)
# {{/docs-fragment image_and_env}}

# {{docs-fragment task}}
@env.task
def main(sequence: str) -> list[str]:
    """Given a sequence, outputs files containing the protein structure
    This requires model weights + gpus + large database on aws fsx lustre
    """
    print(f"Running AlphaFold2 for sequence: {sequence}")
    msa = run_msa(sequence)
    print(f"MSA result: {msa}, passing to fold task")
    results = run_fold(sequence, msa)
    print(f"Fold results: {results}")
    return results
# {{/docs-fragment task}}

# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, "AAGGTTCCAA")
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment run}}
