from flytekit import Resources, ImageSpec

from union.app import App

img = ImageSpec(
    name="marimo",
    builder="unionai",
    packages=[
        "marimo",
        "union-runtime",
        "union",
        "uv",
    ],
)

marimo = App(
    name="marimo-union-demo",
    container_image=img,
    args=[
        "marimo",
        "edit",
        "union_demo.py",
        "--no-token",
        "--sandbox",
    ],
    include=["union_demo.py"],
    port=2718,
    limits=Resources(cpu="1", mem="2Gi"),
    env={
        "UV_CACHE_DIR": "/root",
    }
)
