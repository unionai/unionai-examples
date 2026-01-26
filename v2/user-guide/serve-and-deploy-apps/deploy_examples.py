# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b49",
# ]
# ///

"""Deploy examples for the how-app-deployment-works.md documentation."""

import flyte
import flyte.app
from flyte.remote import App


# {{docs-fragment basic-deploy}}
app_env = flyte.app.AppEnvironment(
    name="my-prod-app",
    # ...
)

if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(app_env)

    # Access deployed apps from deployments
    for deployment in deployments:
        for deployed_env in deployment.envs.values():
            print(f"Deployed: {deployed_env.env.name}")
            print(f"URL: {deployed_env.deployed_app.url}")
# {{/docs-fragment basic-deploy}}


# {{docs-fragment deployment-plan}}
app1_env = flyte.app.AppEnvironment(name="backend", ...)
app2_env = flyte.app.AppEnvironment(name="frontend", depends_on=[app1_env], ...)

# Deploying app2_env will also deploy app1_env
deployments = flyte.deploy(app2_env)

# deployments contains both app1_env and app2_env
assert len(deployments) == 2
# {{/docs-fragment deployment-plan}}


# {{docs-fragment clone-with}}
app_env = flyte.app.AppEnvironment(name="my-app", ...)

if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(
        app_env.clone_with(app_env.name, resources=flyte.Resources(cpu="2", memory="2Gi"))
    )
    for deployment in deployments:
        for deployed_env in deployment.envs.values():
            print(f"Deployed: {deployed_env.env.name}")
            print(f"URL: {deployed_env.deployed_app.url}")
# {{/docs-fragment clone-with}}


# {{docs-fragment activation-deactivation}}
if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(app_env)

    app = App.get(name=app_env.name)

    # deactivate the app
    app.deactivate()

    # activate the app
    app.activate()
# {{/docs-fragment activation-deactivation}}

# {{docs-fragment full-deployment}}
if __name__ == "__main__":
    flyte.init_from_config()
    
    deployments = flyte.deploy(
        app_env,
        dryrun=False,
        version="v1.0.0",
        interactive_mode=False,
        copy_style="loaded_modules",
    )
    
    # Access deployed apps from deployments
    for deployment in deployments:
        for deployed_env in deployment.envs.values():
            app = deployed_env.deployed_app
            print(f"Deployed: {deployed_env.env.name}")
            print(f"URL: {app.url}")

            # Activate the app
            app.activate()
            print(f"Activated: {app.name}")
# {{/docs-fragment full-deployment}}


# {{docs-fragment deployment-status}}
deployments = flyte.deploy(app_env)

for deployment in deployments:
    for deployed_env in deployment.envs.values():
        if hasattr(deployed_env, 'deployed_app'):
            # Access deployed environment
            env = deployed_env.env
            app = deployed_env.deployed_app

            # Access deployment info
            print(f"Name: {env.name}")
            print(f"URL: {app.url}")
            print(f"Status: {app.deployment_status}")
# {{/docs-fragment deployment-status}}
