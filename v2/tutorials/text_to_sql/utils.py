from pathlib import Path

import flyte

env = flyte.TaskEnvironment(
    name="text-to-sql",
    image=flyte.Image.from_debian_base(name="text-to-sql")
    .with_pip_packages(
        "flyte>=2.0.0b52",
        "pandas==2.3.2",
        "llama-index-llms-openai==0.5.4",
        "llama-index-embeddings-openai==0.5.0",
        "sqlalchemy==2.0.43",
        "pyarrow>=21.0.0",
        "litellm==1.76.1",
    )
    .with_source_file(Path("ground_truth.csv"), "/root"),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    resources=flyte.Resources(cpu=1),
    cache="auto",
)
