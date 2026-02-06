# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# ///

"""Serve and deploy examples for the _index.md documentation."""

import flyte
import flyte.app


# {{docs-fragment serve-example}}
app_env = flyte.app.AppEnvironment(
    name="my-app",
    image=flyte.app.Image.from_debian_base().with_pip_packages("streamlit==1.41.1"),
    args=["streamlit", "hello", "--server.port", "8080"],
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.serve(app_env)
    print(f"Served at: {app.url}")
# {{/docs-fragment serve-example}}


# {{docs-fragment deploy-example}}
app_env = flyte.app.AppEnvironment(
    name="my-app",
    image=flyte.app.Image.from_debian_base().with_pip_packages("streamlit==1.41.1"),
    args=["streamlit", "hello", "--server.port", "8080"],
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)

if __name__ == "__main__":
    flyte.init_from_config()
    deployments = flyte.deploy(app_env)
    # Access deployed app URL from the deployment
    for deployed_env in deployments[0].envs.values():
        print(f"Deployed: {deployed_env.deployed_app.url}")
# {{/docs-fragment deploy-example}}
