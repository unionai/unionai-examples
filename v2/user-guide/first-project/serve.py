# /// script
# requires-python = "==3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "streamlit>=1.41.0",
#    "sentence-transformers>=2.2.0",
#    "chromadb>=0.4.0",
# ]
# ///

"""
Quote Search App

A Streamlit app that provides semantic search over quotes using ChromaDB embeddings.
The embeddings are loaded from the embedding pipeline output.
"""

# {{docs-fragment imports}}
import flyte
from flyte.app import AppEnvironment, Parameter, RunOutput
# {{end-fragment}}

# {{docs-fragment app-env}}
# Define the app environment
env = AppEnvironment(
    name="quote-search-app",
    description="Semantic search over quotes using embeddings",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "streamlit>=1.41.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
    ),
    args=["streamlit", "run", "app.py", "--server.port", "8080"],
    port=8080,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    parameters=[
        Parameter(
            name="quotes_db",
            value=RunOutput(task_name="quote-embedding.embedding_pipeline", type="directory"),
            download=True,
            env_var="QUOTES_DB_PATH",
        ),
    ],
    include=["app.py"],
    requires_auth=False,
)
# {{end-fragment}}

# {{docs-fragment main}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy the quote search app
    print("Deploying quote search app...")
    deployment = flyte.serve(env)
    print(f"App deployed at: {deployment.url}")
# {{end-fragment}}
