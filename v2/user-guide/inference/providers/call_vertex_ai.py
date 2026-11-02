# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.12.2",
#    "google-auth[requests]",
#    "openai>=1.60",
# ]
# main = "summarize"
# params = "text='First document...'"
# ///

"""Call Google Vertex AI from a task with a service account."""

# {{docs-fragment task}}
import flyte

BASE_URL = "https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/endpoints/openapi"

env = flyte.TaskEnvironment(
    name="vertex-ai-inference",
    image=flyte.Image.from_debian_base().with_pip_packages("google-auth[requests]", "openai"),
    secrets=flyte.Secret(key="vertex-service-account", mount="/etc/flyte/secrets"),
)


@env.task
async def summarize(text: str) -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    from openai import AsyncOpenAI

    credentials = service_account.Credentials.from_service_account_file(
        "/etc/flyte/secrets/vertex-service-account",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())

    client = AsyncOpenAI(base_url=BASE_URL, api_key=credentials.token)
    response = await client.chat.completions.create(
        model="<model-id>",
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(summarize, text="First document...")
    print(run.url)
# {{/docs-fragment task}}
