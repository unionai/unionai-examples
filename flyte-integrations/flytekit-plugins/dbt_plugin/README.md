# DBT

[dbt](https://www.getdbt.com/) is one of the widely-used data transformation
tools for working with data directly in a data warehouse. It's optimized for
analytics use cases and can be used for business intelligence, operational
analytics, and even machine learning.

The Flytekit `dbt` plugin is a Python module that provides an easy way to
invoke basic `dbt` CLI commands from within a Flyte task. The plugin supports
the commands [dbt run](https://docs.getdbt.com/reference/commands/run),
[dbt test](https://docs.getdbt.com/reference/commands/test), and
[dbt source freshness](https://docs.getdbt.com/reference/commands/source).

## Prerequisities

To use the `dbt` plugin you'll need to install the `flytekitplugins-dbt`
plugin.

> See [the PyPi page here](https://pypi.org/project/flytekitplugins-dbt/).

```
$ pip install flytekitplugins-dbt
```

Then install dbt itself. You will have to install `dbt-core` as well as
the correct adapter for the database that you are accessing.

For example, if you are using a Postgres database you would do:

```
$ pip install dbt-postgres
```

This will install `dbt-core` and `dbt-postgres`, but not any of the other
adapters, `dbt-redshift`, `dbt-snowflake`, or `dbt-bigquery`. See
[the official installation page](https://docs.getdbt.com/docs/get-started/pip-install)
for details.

## Running the Example

We use a Postgres database installed on the cluster and an example project from
dbt, called [jaffle-shop](https://github.com/dbt-labs/jaffle_shop).
To run the example on your local machine, do the following.

> The example below is not designed to run directly in your local
> python environment. It must be run in a Kubernetes cluster, either locally on
> your machine using the `flytectl demo start` command or on a cloud cluster.

Start up the demo cluster on your local machine:

```
$ flytectl demo start
```

Pull the pre-built image for this example:

```
$ docker pull ghcr.io/flyteorg/flytecookbook:dbt_example-latest
```

This image is built using the following `Dockerfile` and contains:

- The `flytekitplugins-dbt` and `dbt-postgres` Python dependencies.
- The `jaffle-shop` example.
- A postgres database.

See the `Dockerfile` alongside this README.

To run this example, copy the code in the **dbt example** below into a file
called `dbt_example.py`, then run it on your local container using the
provided image:

```
$ pyflyte run --remote \
       --image ghcr.io/flyteorg/flytecookbook:dbt_example-latest \
       dbt_plugin/dbt_example.py wf
```

Alternatively, you can clone this repo and run the example directly:

```
$ git clone https://github.com/unionai/unionai-examples
$ cd unionai-examples/flyte-integrations/flytekit-plugins/dbt_plugin
$ pyflyte run --remote \
          --image ghcr.io/flyteorg/flytecookbook:dbt_example-latest \
          dbt_plugin/dbt_example.py wf
```
