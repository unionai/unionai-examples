# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "openai>=1.60",
# ]
# ///

"""Call a deployed llama.cpp app with the OpenAI SDK."""

# {{docs-fragment client}}
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-app-url>/v1",
    api_key="<your-api-key>",
)

response = client.chat.completions.create(
    model="gemma-3-4b-it",  # The app's model_id
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(response.choices[0].message.content)
# {{/docs-fragment client}}
