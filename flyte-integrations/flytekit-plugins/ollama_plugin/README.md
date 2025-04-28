# Ollama

Serve large language models (LLMs) in a Flyte task.

[Ollama](https://ollama.com/) simplifies the process of serving fine-tuned LLMs.
Whether you're generating predictions from a customized model or deploying it across different hardware setups,
Ollama enables you to encapsulate the entire workflow in a single pipeline.

## Installation

To use the Ollama plugin, run the following command:

```shell
$ pip install flytekitplugins-inference
```

> Ollama can only be run in a Flyte cluster (not in local Python or a local demo cluster)
> as it must be deployed as a sidecar service in a Kubernetes pod.
