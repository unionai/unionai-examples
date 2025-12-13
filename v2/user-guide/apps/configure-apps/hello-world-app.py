"""A basic "Hello World" app example with custom subdomain."""

import flyte
import flyte.app

# {{docs-fragment image}}
image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("streamlit==1.41.1")
# {{/docs-fragment image}}

# {{docs-fragment app-env}}
app_env = flyte.app.AppEnvironment(
    name="hello-world-app",
    image=image,
    command="streamlit hello --server.port 8080",
    port=8080,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    requires_auth=False,
    domain=flyte.app.Domain(subdomain="hello"),
)
# {{/docs-fragment app-env}}

# {{docs-fragment deploy}}
if __name__ == "__main__":
    flyte.init_from_config()
    
    # Deploy the app
    app = flyte.deploy(app_env)
    print(f"App deployed at: {app[0].url}")
    print(f"Custom subdomain: https://hello.<your-domain>")
# {{/docs-fragment deploy}}

