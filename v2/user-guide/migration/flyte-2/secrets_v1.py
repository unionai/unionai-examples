from flytekit import task, workflow, Secret, current_context


@task(secret_requests=[Secret(group="openai", key="api_key")])
def call_api() -> str:
    token = current_context().secrets.get(group="openai", key="api_key")
    return f"token has {len(token)} chars"


@workflow
def main() -> str:
    return call_api()
