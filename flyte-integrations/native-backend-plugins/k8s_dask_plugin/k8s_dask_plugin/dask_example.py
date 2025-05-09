# # Running a Dask Task
#
# {{run-on-union}}
#
# The plugin establishes a distinct virtual and short-lived cluster for each Dask task, with Flyte overseeing the entire cluster lifecycle.
#
# To begin, import the required dependencies.
from flytekit import ImageSpec, Resources, task

# Create an `ImageSpec` to encompass all the dependencies needed for the Dask task.
custom_image = ImageSpec(registry="ghcr.io/flyteorg", packages=["flytekitplugins-dask"])

# > [!NOTE]
# > Replace `ghcr.io/flyteorg` with a container registry you've access to publish to.
# > To upload the image to the local registry in the demo cluster, indicate the registry as `localhost:30000`.
#
# The following imports are required to configure the Dask cluster in Flyte.
# You can load them on demand.
if custom_image.is_container():
    from dask import array as da
    from flytekitplugins.dask import Dask, WorkerGroup


# When executed locally, Flyte launches a Dask cluster on the local environment.
# However, when executed remotely, Flyte triggers the creation of a cluster with a size determined by the
# specified `flytekitplugins.dask.Dask` configuration.
@task(
    task_config=Dask(
        workers=WorkerGroup(
            number_of_workers=10,
            limits=Resources(cpu="4", mem="10Gi"),
        ),
    ),
    limits=Resources(cpu="1", mem="2Gi"),
    container_image=custom_image,
)
def hello_dask(size: int) -> float:
    # Dask will automatically generate a client in the background using the Client() function.
    # When running remotely, the Client() function will utilize the deployed Dask cluster.
    array = da.random.random(size)
    return float(array.mean().compute())


# You can execute the task locally as follows:
if __name__ == "__main__":
    print(hello_dask(size=1000))
