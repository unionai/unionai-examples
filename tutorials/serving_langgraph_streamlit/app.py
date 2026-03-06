from union import ImageSpec, Resources
from union.app import App

image = ImageSpec(
    name="streamlit-app",
    builder="unionai",
    requirements="uv.lock",
)

app1 = App(
    name="streamlit-langgraph",
    container_image=image,
    args="streamlit run conv.py --server.port 8080",
    port=8080,
    include=["conv.py"],
    limits=Resources(cpu="2", mem="24Gi"),
    requests=Resources(cpu="2", mem="24Gi", ephemeral_storage="40Gi"),
)