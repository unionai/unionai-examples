# TensorFlow Distributed

TensorFlow operator is useful to natively run distributed TensorFlow training jobs on Flyte.
It leverages the [Kubeflow training operator](https://github.com/kubeflow/training-operator).

## Install the plugin

To install the Kubeflow TensorFlow plugin, run the following command:

```shell
$ pip install flytekitplugins-kftensorflow
```

To enable the plugin in the backend, follow instructions outlined in the [Kubernetes plugins](https://www.union.ai/docs/flyte/deployment/flyte-plugins/kubernetes-plugins/) section of the Flyte docs.

## Run the example on the Flyte cluster

To run the provided example on the Flyte cluster, use the following command:

```shell
$ pyflyte run --remote tf_mnist.py \
  mnist_tensorflow_workflow
```
