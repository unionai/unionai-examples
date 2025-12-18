# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
#    "fastapi",
#    "uvicorn",
# ]
# ///

"""A simple "Hello World" FastAPI app example for serving."""

from fastapi import FastAPI
import pathlib
import flyte
from flyte.app.extras import FastAPIAppEnvironment

# Define a simple FastAPI application
app = FastAPI(
    title="Hello World API",
    description="A simple FastAPI application",
    version="1.0.0",
)

# Create an AppEnvironment for the FastAPI app
env = FastAPIAppEnvironment(
    name="hello-app",
    app=app,
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi",
        "uvicorn",
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

# Define API endpoints
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Serving this script will deploy and serve the app on your Union/Flyte instance.
if __name__ == "__main__":
    # Initialize Flyte from a config file.
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

    # Serve the app remotely.
    app_instance = flyte.serve(env)

    # Print the app URL.
    print(app_instance.url)
    print("App 'hello-app' is now serving.")

