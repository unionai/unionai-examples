# Ray

[KubeRay](https://github.com/ray-project/kuberay) is an open-source toolkit designed to facilitate the execution of
Ray applications on Kubernetes. It offers a range of tools that enhance the operational aspects of
running and overseeing Ray on Kubernetes.

Key components include:

- Ray Operator
- Backend services for cluster resource creation and deletion
- Kubectl plugin/CLI for CRD object management
- Seamless integration of Jobs and Serving functionality with Clusters

## Install the plugin

To install the Ray plugin, run the following command:

```bash
pip install flytekitplugins-ray
```

To enable the plugin in the backend, refer to the instructions provided in the [Kubernetes plugins](https://www.union.ai/docs/flyte/deployment/flyte-plugins/kubernetes-plugins/) section of the Flyte docs.

## Implementation details

### Submit a Ray job to existing cluster

```python
import ray
from flytekit import task
from flytekitplugins.ray import RayJobConfig
@ray.remote
def f(x):
    return x * x
@task(
    task_config=RayJobConfig(
        address=<RAY_CLUSTER_ADDRESS>
        runtime_env={"pip": ["numpy", "pandas"]}
    )
)
def ray_task() -> typing.List[int]:
    futures = [f.remote(i) for i in range(5)]
    return ray.get(futures)
```

### Create a Ray cluster managed by Flyte and run a Ray Job on the cluster

```python
import ray
from flytekit import task
from flytekitplugins.ray import RayJobConfig, WorkerNodeConfig, HeadNodeConfig
@task(task_config=RayJobConfig(worker_node_config=[WorkerNodeConfig(group_name="test-group", replicas=10)]))
def ray_task() -> typing.List[int]:
    futures = [f.remote(i) for i in range(5)]
    return ray.get(futures)
```

## Run the example on the Flyte cluster

To run the provided example on the Flyte cluster, use the following command:

```bash
pyflyte run --remote ray_example.py \
    ray_workflow --n 10
```
