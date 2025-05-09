# # Run Ollama on Union Serving platform
#
# In this example, we use Union Serving to Ollama and serve DeepSeek R1 Distilled Qwen 1.58b Model

# ## Caching the model from hugging face
# In order to speed up Ollama startup. We will cache the model from Hugging Face. This is
# important because it allows us to avoid downloading the model every time we start
# the Ollama service. Caching the model will significantly reduce the startup time
# for the Ollama service. This is especially important if we are deploying the
# application to a production environment where we need to ensure that the service
# starts up quickly and efficiently.

# To cache the model, we will use the `download_model` task defined in the `model_wf.py`
# file. This task will download the specified model from Hugging Face and publish
# it as an artifact. The `download_model` task is decorated with the `@task` decorator,
# which allows us to define a task that can be executed as part of a workflow. The
# `container_image` parameter specifies the container image to use for the task.
# The `cache=True` parameter indicates that we want to cache the results of the
# task. The `cache_version` parameter specifies the version of the cache. This is
# important because it allows us to invalidate the cache if we make changes to the
# task or if we want to download a different version of the model. In this example,
# we are using version "1.0" for the cache. The `download_model` task takes a single
# parameter, `model`, which specifies the name of the model to download. The task
# will download the model from Hugging Face and save it to a directory. The task
# will then publish the directory as an artifact. This artifact can be used in
# subsequent tasks or workflows. The `MyOllamaModel` class is used to define the
# input for the task. The `MyOllamaModel` class is a subclass of `Artifact` that
# represents the model that will be downloaded. The `partition_keys` parameter
# specifies the keys that will be used to partition the artifact. In this case, we
# are using the `model` key to partition the artifact. This allows us to download
# different versions of the model by specifying different values for the `model`
# parameter. The `download_model` task will download the specified model and save
# it to a directory. The directory will be published as an artifact. This artifact
# can then be referenced in our application using the `MyOllamaModel` class.
# The `MyOllamaModel` class is used to reference the downloaded model in our
# application.
# The `query()` method of the `MyOllamaModel` class will return the path to the
# downloaded model. This path can be used to reference the model in our application.
# 
# Once the model is cached, we can use it in our application. The `Artifact` class
# provides a way to reference the cached model. We can use the `query()` method to
# get the path to the cached model. This is useful when we want to use the model in
# our application. The `query()` method will return the path to the cached model, which
# can be used to reference the model in our application.
#
# ## Defining the Application
# 
# Now that we have the model cached, we can define our application. We will use the
# `Artifact` class to reference the cached model. We will also define the application
# configuration, including the container image, the input for the model, and the
# resources required for the application. We will also include the necessary arguments
# to start the application.


# Import the necessary classes from the union package
from union import Artifact, Resources, ImageSpec
from union.app import App, Input
import os
import model_wf


# We will define the application configuration. This includes the name of the
# application, the inputs required for the application, the arguments to start
# the application, the container image to use, the port to expose, the resource
# limits for the application, and the environment variables required for the
# application to run. We will also set the `OLLAMA_MODELS` environment variable
# to the path where the model will be downloaded to. This is important because
# the Ollama model needs to be accessible to the application when it starts up.
# file: app.py

MODEL = os.environ.get("MODEL", "llama3.1")

ollama_app = App(
    name="ollama-serve",
    inputs=[
        Input(
            name="my_model",
            value=model_wf.MyOllamaModel.query(model="llama3.1"),
            download=True,
            mount="/home/union/.ollama/models",
        )
    ],
    container_image=model_wf.image.with_packages(["union-runtime"]),
    args=[
        "ollama",
        "serve",
    ],
    limits=Resources(cpu="2", mem="8Gi", ephemeral_storage="40Gi"),
    requests=Resources(ephemeral_storage="40Gi"),
    port=11434,
    min_replicas=1,
    env={
        "UNION_SDK_LOGGING_LEVEL": "10",
        "OLLAMA_HOST": "0.0.0.0",
        "OLLAMA_ORIGINS": "*",
    },
)

# Finally, we can deploy the application. The `deploy` method will package the
# application and deploy it to the Union Serving platform. This will include
# building the container image, packaging the application, and deploying it to
# the Union Serving platform. Once the deployment is complete, we will see
# a message indicating that the application has been successfully deployed.

# ```bash
# union deploy apps app.py ollama-serve
# ```
