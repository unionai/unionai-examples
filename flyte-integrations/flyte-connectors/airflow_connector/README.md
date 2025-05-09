# Airflow connector

[Apache Airflow](https://airflow.apache.org) is a widely used open source platform for managing workflows with a robust ecosystem. Flyte provides an Airflow plugin that allows you to run Airflow tasks as Flyte tasks.
This allows you to use the Airflow plugin ecosystem in conjunction with Flyte's powerful task execution and orchestration capabilities.

> The Airflow connector does not support all [Airflow operators](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html).
> We have tested many, but if you run into issues, please >[file a bug report](https://github.com/flyteorg/flyte/issues/new?assignees=&labels=bug%2Cuntriaged&projects=&template=bug_report.yaml&title=%5BBUG%5D+).

## Installation

To install the plugin, run the following command:

```shell
$ pip install flytekitplugins-airflow
```

This plugin has two components:
* **Airflow compiler:** This component compiles Airflow tasks to Flyte tasks, so Airflow tasks can be directly used inside the Flyte workflow.
* **Airflow connector:** This component allows you to execute Airflow tasks either locally or on a Flyte cluster.
