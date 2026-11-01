# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "openai>=1.60",
#    "fastapi",
#    "uvicorn",
# ]
# ///

"""Call Fireworks AI from a FastAPI app with your own API key."""

import os

import flyte
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment
from openai import AsyncOpenAI

app = FastAPI()


@app.post("/summarize")
async def summarize(text: str) -> dict:
    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    response = await client.chat.completions.create(
        model="accounts/fireworks/models/glm-5p3-flash",
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return {"summary": response.choices[0].message.content}


# {{docs-fragment app-env}}
app_env = FastAPIAppEnvironment(
    name="summarizer-api",
    app=app,
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn", "openai"),
    secrets=flyte.Secret(key="fireworks-api-key", as_env_var="FIREWORKS_API_KEY"),
)
# {{/docs-fragment app-env}}

if __name__ == "__main__":
    flyte.init_from_config()
    served = flyte.serve(app_env)
    print(served.url)
