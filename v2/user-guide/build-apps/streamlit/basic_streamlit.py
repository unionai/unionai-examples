# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b45",
# ]
# ///

"""A basic Streamlit app using the built-in hello demo."""

import flyte
import flyte.app

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1")
# {{/docs-fragment image}}

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="streamlit-hello",
    image=image,
    command="streamlit hello --server.port 8080",
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.deploy(app_env)
    print(f"Deployed app: {app[0].summary_repr()}")
# {{/docs-fragment deploy}}

