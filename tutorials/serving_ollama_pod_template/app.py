from union.app import App, Input, ScalingMetric
from union import ImageSpec, Artifact, Resources
from flytekit.extras.accelerators import L4

MoonDreamArtifact = Artifact(name="ollama-moondream")

ollama_image = ImageSpec(
    name="ollama-serve",
    apt_packages=["curl"],
    packages=["union-runtime>=0.1.11"],
    registry="ghcr.io/unionai-oss",
    commands=[
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
    ],
)

ollama = App(
    name="ollama-serve",
    inputs=[
        Input(
            value=MoonDreamArtifact.query(),
            mount="/home/.ollama",
        )
    ],
    container_image=ollama_image,
    port=11434,
    args="ollama serve",
    limits=Resources(cpu="8", mem="10Gi", ephemeral_storage="20Gi", gpu="1"),
    accelerator=L4,
    env={
        "OLLAMA_HOST": "0.0.0.0",
        "OLLAMA_ORIGINS": "*",
        "OLLAMA_MODELS": "/home/.ollama/models",
    },
    min_replicas=0,
    max_replicas=1,
    scaledown_after=200,
)


streamlit_image = ImageSpec(
    name="streamlit-chat",
    packages=["streamlit==1.41.1", "union-runtime>=0.1.11", "ollama==0.4.7"],
    registry="ghcr.io/unionai-oss",
)

streamlit_app = App(
    name="ollama-streamlit",
    inputs=[
        Input(
            value=ollama.query_endpoint(),
            env_var="OLLAMA_ENDPOINT",
        )
    ],
    limits=Resources(cpu="2", mem="4Gi"),
    container_image=streamlit_image,
    port=8082,
    include=["./streamlit_app.py"],
    args=[
        "streamlit",
        "run",
        "streamlit_app.py",
        "--server.port",
        "8082",
    ],
    min_replicas=0,
    max_replicas=1,
)
