# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "streamlit",
#    "openai",
# ]
# ///

"""An Ollama chat app fronted by a Streamlit UI.

Ollama runs internally as an OpenAI-compatible server; a Streamlit chat
interface fronts it. Only the Streamlit port is exposed to the platform — the
browser talks to Streamlit, and Streamlit talks to Ollama over localhost.
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import flyte
import flyte.app

MODEL = "qwen3:0.6b"
OLLAMA_PORT = 11434  # internal only, not exposed
APP_PORT = 8080  # Streamlit UI, exposed to the platform

file_name = Path(__file__).name

# {{docs-fragment app-env}}
image = (
    flyte.Image.from_debian_base(python_version=(3, 12))
    .with_apt_packages("curl")
    .with_commands(["curl -fsSL https://ollama.com/install.sh | sh"])
    .with_pip_packages("streamlit==1.41.1", "openai")
)

app_env = flyte.app.AppEnvironment(
    name="ollama-streamlit",
    image=image,
    args=["python", file_name, "--server"],
    port=APP_PORT,
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L40s:1", disk="20Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
    requires_auth=False,
)
# {{/docs-fragment app-env}}


# {{docs-fragment ui}}
def render_ui() -> None:
    """The Streamlit chat UI. Talks to the local Ollama OpenAI-compatible API."""
    import streamlit as st
    from openai import OpenAI

    st.set_page_config(page_title="Ollama Chat", page_icon="🦙")
    st.title("🦙 Ollama Chat")

    client = OpenAI(base_url=f"http://127.0.0.1:{OLLAMA_PORT}/v1", api_key="ollama")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = client.chat.completions.create(
            model=MODEL, messages=st.session_state.messages
        )
        answer = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
# {{/docs-fragment ui}}


# {{docs-fragment server}}
def start_ollama() -> None:
    """Launch `ollama serve` in the background and pull the model."""
    subprocess.Popen(
        ["ollama", "serve"],
        env={**os.environ, "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}"},
    )
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/version", timeout=2)
            break
        except Exception:
            time.sleep(1)
    subprocess.run(
        ["ollama", "pull", MODEL],
        env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{OLLAMA_PORT}"},
        check=True,
    )
# {{/docs-fragment server}}


# {{docs-fragment deploy}}
if __name__ == "__main__":
    if "--ui" in sys.argv:
        # Re-entry: Streamlit runs this file to render the UI.
        render_ui()
    elif "--server" in sys.argv:
        # Container entrypoint: start Ollama, then hand the port to Streamlit.
        start_ollama()
        subprocess.run(
            [
                "streamlit", "run", file_name,
                "--server.port", str(APP_PORT),
                "--server.address", "0.0.0.0",
                "--", "--ui",
            ],
            check=True,
        )
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        app = flyte.serve(app_env)
        print(f"Deployed app: {app.url}")
# {{/docs-fragment deploy}}
