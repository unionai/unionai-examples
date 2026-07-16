# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# ///

"""A simple Ollama serving app example.

Unlike vLLM and SGLang, Ollama has no dedicated ``*AppEnvironment`` plugin, so it
is served with the generic ``flyte.app.AppEnvironment``: an image that installs
Ollama plus a small ``--server`` entrypoint that launches ``ollama serve`` and
pulls the model on startup. Ollama exposes an OpenAI-compatible API, so clients
call it exactly like the vLLM / SGLang apps.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import flyte
import flyte.app

# Any tag from https://ollama.com/library. Small models run fine on CPU; larger
# ones benefit from the GPU requested below.
MODEL = "qwen3:0.6b"

# Bind Ollama to the app port so the platform can route to it.
PORT = 8080

file_name = Path(__file__).name

# {{docs-fragment ollama-image}}
# Install Ollama on top of the default Flyte base image. `install.sh` drops the
# `ollama` binary into /usr/local/bin (no systemd is needed inside a container).
image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("curl")
    .with_commands(["curl -fsSL https://ollama.com/install.sh | sh"])
)
# {{/docs-fragment ollama-image}}

# {{docs-fragment ollama-app}}
ollama_app = flyte.app.AppEnvironment(
    name="ollama-app",
    image=image,
    args=["python", file_name, "--server"],
    port=PORT,
    resources=flyte.Resources(
        cpu="4",
        memory="16Gi",
        gpu="L40s:1",  # GPU accelerates inference; omit to run small models on CPU
        disk="20Gi",
    ),
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,  # Scale down after 5 minutes of inactivity
    ),
    requires_auth=False,
)
# {{/docs-fragment ollama-app}}


# {{docs-fragment server}}
def serve() -> None:
    """Start `ollama serve`, wait for it, pull the model, then block."""
    # Bind to all interfaces so the platform can route to the server.
    server = subprocess.Popen(
        ["ollama", "serve"],
        env={**os.environ, "OLLAMA_HOST": f"0.0.0.0:{PORT}"},
    )

    # Wait for the server to accept connections.
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/version", timeout=2)
            break
        except Exception:
            time.sleep(1)

    # Pull the model so the OpenAI-compatible endpoint can serve it.
    subprocess.run(
        ["ollama", "pull", MODEL],
        env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{PORT}"},
        check=True,
    )
    print(f"Ollama serving '{MODEL}' on port {PORT}")

    server.wait()
# {{/docs-fragment server}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    if "--server" in sys.argv:
        serve()
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        app = flyte.serve(ollama_app)
        print(f"Deployed Ollama app: {app.url}")
# {{/docs-fragment deploy}}
