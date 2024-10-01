import json
import os
import subprocess
from pathlib import Path


def ios_local_deployment(
    model_hf_url: str, bundle_weight: bool, model_id: str, estimated_vram_bytes
):
    if not os.path.exists("./mlc-llm"):
        subprocess.run(
            ["git", "clone", "https://github.com/mlc-ai/mlc-llm.git", "./mlc-llm"],
            check=True,
        )

        # Navigate to the repo and update submodules
        subprocess.run(
            ["git", "-C", "mlc-llm", "submodule", "update", "--init", "--recursive"],
            check=True,
        )

    # Update the JSON structure
    new_model_entry = {
        "device": "iphone",
        "model_list": [
            {
                "model": model_hf_url,
                "model_id": model_id,
                "estimated_vram_bytes": estimated_vram_bytes,
                "bundle_weight": bundle_weight,
            }
        ],
    }

    json_file_path = Path("./mlc-llm/ios/MLCChat/mlc-package-config.json")
    with open(json_file_path, "w") as f:
        json.dump(new_model_entry, f, indent=4)

    mlc_chat_dir = Path("./mlc-llm/ios/MLCChat")
    mlc_llm_source_dir = Path(
        "<ROOT>/mlc-llm"
    ) # give absolute path

    subprocess.run(
        ["python", "-m", "mlc_llm", "package"],
        cwd=mlc_chat_dir,
        env={**os.environ, "MLC_LLM_SOURCE_DIR": mlc_llm_source_dir},
        check=True,
    )


if __name__ == "__main__":
    ios_local_deployment(
        model_hf_url="<HF_REPO_URL>",
        bundle_weight=True,
        model_id="Llama-3-8B-Instruct-v0.1-q4f16_1",
        estimated_vram_bytes=3316000000,
    )
