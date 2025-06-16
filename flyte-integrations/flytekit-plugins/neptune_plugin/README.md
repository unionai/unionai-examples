# Neptune Scale

[Neptune](https://neptune.ai/) is an experiment tracker for large-scale model training. It allows AI researchers to monitor their model training in real time, visualize and compare experiments, and collaborate on them with a team. This plugin enables seamless use of Neptune within Flyte by configuring links between the two platforms. You can find more information about how to use Neptune in their [documentation](https://docs.neptune.ai/).

## Installation

To install the Flyte Neptune plugin, run the following command:

```shell
$ pip install flytekitplugins-neptune
```

## Local testing

To run {doc}`Neptune example <neptune_example>` locally:

1. Neptune Scale is available to select customers. You can access it [here](https://neptune.ai/free-trial).
2. Create a project on Neptune.
3. In the example, set `NEPTUNE_PROJECT` to your project name.
4. Add a secret using [Flyte's Secrets manager](https://www.union.ai/docs/flyte/deployment/flyte-configuration/secrets) with `key="neptune-api-token"` and `group="neptune-api-group"`
5. If you want to see the dynamic log links in the UI, then add the configuration available in the next section.

## Flyte deployment configuration

To enable dynamic log links, add the plugin to Flyte's configuration file as follows:

```yaml
plugins:
  logs:
    dynamic-log-links:
      - neptune-scale-run:
          displayName: Neptune Run
          templateUris:
            - "https://scale.neptune.ai/{{ .taskConfig.project }}/-/run/?customId={{ .podName }}"
      - neptune-scale-custom-id:
          displayName: Neptune Run
          templateUris:
            - "https://scale.neptune.ai/{{ .taskConfig.project }}/-/run/?customId={{ .taskConfig.id }}"
```
