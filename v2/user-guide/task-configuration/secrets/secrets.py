# {{docs-fragment literal}}
env = flyte.TaskEnvironment(
    name="my_task_env",
    secrets=[
        flyte.Secret(key="MY_SECRET_KEY", as_env_var="MY_SECRET_ENV_VAR"),
    ]
)

@env.task
def t1():
    my_secret_value = os.getenv("MY_SECRET_ENV_VAR")
    # Do something with the secret
# {{/docs-fragment literal}}

# {{docs-fragment file}}
env = flyte.TaskEnvironment(
    name="my_task_env",
    secrets=[
        flyte.Secret(key="MY_SECRET_KEY", mount="/etc/flyte/secrets"),
    ]
)

@env.task
def t1():
    with open("/etc/flyte/secrets/MY_SECRET_KEY", "r") as f:
        my_secret_file_content = f.read()
        # Do something with the secret file content
# {{/docs-fragment file}}
