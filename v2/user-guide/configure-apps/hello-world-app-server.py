# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# ///

import flyte
import flyte.app

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1")
# {{/docs-fragment image}}

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="hello-world-app-server",
    image=image,
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
    domain=flyte.app.Domain(subdomain="hello-server"),
)

@app_env.server
def server():
    import subprocess
    subprocess.run(["streamlit", "hello", "--server.port", "8080"], check=False)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy the app
    app = flyte.serve(app_env)
    print(f"App served at: {app.url}")
# {{/docs-fragment deploy}}
