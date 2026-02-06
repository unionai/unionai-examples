# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# ///

"""A basic Streamlit app using the built-in hello demo."""

# {{docs-fragment app-definition}}
import flyte
import flyte.app

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1")

app_env = flyte.app.AppEnvironment(
    name="streamlit-hello",
    image=image,
    args="streamlit hello --server.port 8080",
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
)

if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.deploy(app_env)
    print(f"Deployed app: {app[0].summary_repr()}")
# {{/docs-fragment app-definition}}
