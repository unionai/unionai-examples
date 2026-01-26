# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b49",
# ]
# ///

"""Activation examples for the activating-and-deactivating-apps.md documentation."""

import flyte
import flyte.app
from flyte.remote import App


app_env = flyte.app.AppEnvironment(
    name="my-app",
    # ...
)


# {{docs-fragment activate-after-deployment}}
# Deploy the app
deployments = flyte.deploy(app_env)

# Activate the app
app = App.get(name=app_env.name)
app.activate()

print(f"Activated app: {app.name}")
print(f"URL: {app.url}")
# {{/docs-fragment activate-after-deployment}}


# {{docs-fragment activate-app}}
app = App.get(name="my-app")
app.activate()
# {{/docs-fragment activate-app}}


# {{docs-fragment check-activation-status}}
app = App.get(name="my-app")
print(f"Active: {app.is_active()}")
print(f"Revision: {app.revision}")
# {{/docs-fragment check-activation-status}}


# {{docs-fragment deactivation}}
app = App.get(name="my-app")
app.deactivate()

print(f"Deactivated app: {app.name}")
# {{/docs-fragment deactivation}}


# {{docs-fragment typical-deployment-workflow}}
# 1. Deploy new version
deployments = flyte.deploy(
    app_env,
    version="v2.0.0",
)

# 2. Get the deployed app
new_app = App.get(name="my-app")
# Test endpoints, etc.

# 3. Activate the new version
new_app.activate()

print(f"Deployed and activated version {new_app.revision}")
# {{/docs-fragment typical-deployment-workflow}}


# {{docs-fragment blue-green-deployment}}
# Deploy new version without deactivating old
new_deployments = flyte.deploy(
    app_env,
    version="v2.0.0",
)

new_app = App.get(name="my-app")

# Test new version
# ... testing ...

# Switch traffic to new version
new_app.activate()

print(f"Activated revision {new_app.revision}")
# {{/docs-fragment blue-green-deployment}}


# {{docs-fragment automatic-activation}}
# Automatically activated
app = flyte.serve(app_env)
print(f"Active: {app.is_active()}")  # True
# {{/docs-fragment automatic-activation}}


# {{docs-fragment complete-example}}
app_env = flyte.app.AppEnvironment(
    name="my-prod-app",
    # ... configuration ...
)

if __name__ == "__main__":
    flyte.init_from_config()
    
    # Deploy
    deployments = flyte.deploy(
        app_env,
        version="v1.0.0",
        project="my-project",
        domain="production",
    )
    
    # Get the deployed app
    app = App.get(name="my-prod-app")
    
    # Activate
    app.activate()
    
    print(f"Deployed and activated: {app.name}")
    print(f"Revision: {app.revision}")
    print(f"URL: {app.url}")
    print(f"Active: {app.is_active()}")
# {{/docs-fragment complete-example}}
