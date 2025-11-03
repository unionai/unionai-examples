# # Deploy Custom Webhooks to Launch Workflows
#
# In this example, we use Union Serving to deploy a custom webhook that can trigger
# any of your registered workflows. We will learn how to use a simple `cURL` command with
# authentication to easily start your workflows!

# {{run-on-union}}

# ## Defining the Application Configuration
#
# First, we define the image spec for the runtime image. We will use `fastapi` for defining
# the serving API.

import os
from union import ImageSpec, Resources, Secret
from union.app import App

image_spec = ImageSpec(
    name="webhook-serving",
    builder="union",
    packages=["union-runtime>=0.1.11", "fastapi[standard]==0.115.11", "union>=0.1.150"],
    registry=os.getenv("IMAGE_SPEC_REGISTRY"),
)

# We define the configuration for for the webhook. It includes `./main.py` which
# defines the FastAPI application. There are also two secrets `WEBHOOK_API_KEY` and
# `MY_UNION_API_KEY`. The `WEBHOOK_API_KEY` is used by the FastAPI app to authenticate
# the webhook and `MY_UNION_API_KEY` is used to authenticate `UnionRemote` with `Union`.

app = App(
    name="fastapi-webhook",
    container_image=image_spec,
    limits=Resources(cpu="1", mem="1Gi"),
    port=8080,
    include=["./main.py"],
    args="fastapi run --port 8080",
    secrets=[
        Secret(key="WEBHOOK_API_KEY", env_var="WEBHOOK_API_KEY"),
        Secret(key="MY_UNION_API_KEY", env_var="UNION_API_KEY"),
    ],
    requires_auth=False,
)

# With `requires_auth=False`, the endpoint can be reached without going through Union's
# authentication, which is okay since we are rolling our own `WEBHOOK_API_KEY`. Before
# we can deploy the app, we create the secrets required by the application:
#
# ```shell
# $ union create secret --name WEBHOOK_API_KEY
# ````
#
# For this example, we'll assume that `WEBHOOK_API_KEY` is defined in your shell.
# Next, to create the `MY_UNION_API_KEY` secret, we need to first create a admin api-key:
#
# ```shell
# $ union create api-key admin --name admin-union-api-key
# ```
#
# You will see a `export UNION_API_KEY=<api-key>`, copy the api key and create a secret
# with it:
#
# ```shell
# $ union create secret --name MY_UNION_API_KEY
# ```
#
# Finally, you can now deploy the application:
#
# ```shell
# $ union deploy apps app.py fastapi-webhook
# ```
#
# Deploying the application will stream the status:
#
# ```console
# Image ghcr.io/thomasjpfan/webhook-serving:KXwIrIyoU_Decb0wgPy23A found. Skip building.
# ✨ Deploying Application: fastapi-webhook
# 🔎 Console URL: https://<union-tenant>/console/projects/thomasjpfan/domains/development/apps/fastapi-webhook
# [Status] Pending: App is pending deployment
# [Status] Pending: RevisionMissing: Configuration "fastapi-webhook" is waiting for a Revision to become ready.
# [Status] Pending: IngressNotConfigured: Ingress has not yet been reconciled.
# [Status] Pending: Uninitialized: Waiting for load balancer to be ready
# [Status] Started: Service is ready
#
# 🚀 Deployed Endpoint: https://rough-meadow-97cf5.apps.<union-tenant>
# ```
#
# Save the deployed endpoint for the next section.
#
# ## Launching the Workflow with Webhook
#
# For this demo, we'll register a simple workflow `add_one`:
#
# ```shell
# $ union register wf.py
# ```
#
# This command outputs the version of the workflow:
#
# ```console
# Computed version is Kh8OaZYZzsiLipGwTS18rw
# Serializing and registering 3 flyte entities
# [✔] Task: wf.add_one
# [✔] Workflow: wf.add_one_wf
# [✔] Launch Plan: wf.add_one_wf
# Successfully registered 3 entities
# ```
#
# Finally, we can launch the workflow with our application with curl:
#
# ```shell
# $ export WEBHOOK_API_KEY=...  # use your custom api key
# $ export APP_ENDPOINT=... # Use your app endpoint
# $ export WF_VERSION=... # Use your workflow version
# $ curl -X 'POST' \
#   'https://$APP_ENDPOINT/run-workflow/thomasjpfan/development/wf.add_one_wf/$WF_VERSION' \
#   -H 'accept: application/json' \
#   -H "Authorization: Bearer $WEBHOOK_API_KEY" \
#   -H 'Content-Type: application/json' \
#   -d '{"x": 2}'
# ```
#
# The response will contain the URL of the execution:
#
# ```console
# {"url": "https://<union-tenant>/..."}
# ```
#
# You can modify `main.py` to make adjustments to the FastAPI webhook for your use case.
#
# ## FastAPI code
#
# You can find the FastAPI code in the `main.py` file [here](https://github.com/unionai/unionai-examples/blob/main/v1/tutorials/serving_webhook/main.py).
# With the above App configuration, the `WEBHOOK_API_KEY` environment variable is injected into the FastAPI runtime.
#
