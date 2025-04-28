# NIM

Serve optimized model containers with NIM in a Flyte task.

[NVIDIA NIM](https://www.nvidia.com/en-in/ai/), part of NVIDIA AI Enterprise, provides a streamlined path
for developing AI-powered enterprise applications and deploying AI models in production.
It includes an out-of-the-box optimization suite, enabling AI model deployment across any cloud,
data center, or workstation. Since NIM can be self-hosted, there is greater control over cost, data privacy,
and more visibility into behind-the-scenes operations.

With NIM, you can invoke the model's endpoint as if it is hosted locally, minimizing network overhead.

## Installation

To use the NIM plugin, run the following command:

```bash
pip install flytekitplugins-inference
```

> NIM can only be run in a Flyte cluster (not in local python or on local the demo cluster)
> as it must be deployed as a sidecar service in a Kubernetes pod.
