"""Utility functions for the agentic RAG tutorial."""

import os
from functools import wraps, partial

from flytekit import current_context


def env_secret(fn=None, *, secret_name: str, env_var: str):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        os.environ[env_var] = current_context().secrets.get(key=secret_name)
        return fn(*args, **kwargs)

    if fn is None:
        return partial(env_secret, secret_name=secret_name, env_var=env_var)

    return wrapper


openai_env_secret = partial(
    env_secret,
    secret_name="openai_api_key",
    env_var="OPENAI_API_KEY",
)
