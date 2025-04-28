# Kubernetes Pods

> This plugin is no longer needed and is here only for backwards compatibility.
> No new versions will be published after v1.13.x.
> Please use the `pod_template` and `pod_template_name` arguments to `@task` as described in
> [Configuration task pods with K8S PodTemplates](https://www.union.ai/docs/flyte/deployment/flyte-configuration/configuring-podtemplates/) instead.

Flyte tasks, represented by the `@task` decorator, are essentially single functions that run in one container.
However, there may be situations where you need to run a job with more than one container or require additional capabilities, such as:

- Running a hyper-parameter optimizer that stores state in a Redis database
- Simulating a service locally
- Running a sidecar container for logging and monitoring purposes
- Running a pod with additional capabilities, such as mounting volumes

To support these use cases, Flyte provides a Pod configuration that allows you to customize the pod specification used to run the task.
This simplifies the process of implementing the Kubernetes pod abstraction for running multiple containers.

> A Kubernetes pod will not exit if it contains any sidecar containers (containers that do not exit automatically).
> You do not need to write any additional code to handle this, as Flyte automatically manages pod tasks.

## Installation

To use the Flytekit pod plugin, run the following command:

```bash
pip install flytekitplugins-pod
```