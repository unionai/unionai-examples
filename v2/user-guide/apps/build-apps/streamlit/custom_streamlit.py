"""A custom Streamlit app with multiple files."""

import flyte
import flyte.app

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "streamlit==1.41.1",
    "pandas==2.2.3",
    "numpy==2.2.3",
)
# {{/docs-fragment image}}

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="streamlit-custom-app",
    image=image,
    args="streamlit run main.py --server.port 8080",
    port=8080,
    include=["main.py", "utils.py"],  # Include your app files
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    app = flyte.deploy(app_env)
    print(f"App URL: {app[0].url}")
# {{/docs-fragment deploy}}

