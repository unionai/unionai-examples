# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b25",
#    "flyteplugins-connectors[bigquery]",
#    "pandas"
# ]
# main = "full_bigquery_wf"
# params = "1"
# ///

# # BigQuery connector – templated query + DataFrame example usage
#
# {{run-on-union}}
#
# This example shows how to use a Flyte BigQueryTask to execute a query.

import flyte
import pandas as pd
from flyte.io import DataFrame
from flyteplugins.connectors.bigquery.task import BigQueryConfig, BigQueryTask
from typing_extensions import Annotated

# This is the world's simplest query. Note that in order for deployment to work properly, you'll need to give your
# BigQuery task a name that's unique across your project/domain for your Flyte installation.
bigquery_task_no_io = BigQueryTask(
    name="sql.bigquery.no_io",
    inputs={},
    output_dataframe_type=DataFrame,
    query_template="SELECT 1",
    plugin_config=BigQueryConfig(ProjectID="flyte"),
)

flyte.TaskEnvironment.from_task("bigquery_task_no_io_env", bigquery_task_no_io)


# Of course, in real world applications we are usually more interested in using BigQuery to query a dataset.
# In this case we use crypto_dogecoin data which is public dataset in BigQuery
# [here](https://console.cloud.google.com/bigquery?project=bigquery-public-data&page=table&d=crypto_dogecoin&p=bigquery-public-data&t=transactions).
#
# Let's look out how we can parameterize our query to filter results for a specific transaction version, provided as a user input
# specifying a version.

DogeCoinDataset = Annotated[DataFrame, {"hash": str, "size": int, "block_number": int}]

bigquery_task_templatized_query = BigQueryTask(
    name="sql.bigquery.w_io",
    # Define inputs as well as their types that can be used to customize the query.
    inputs={"version": int},
    output_dataframe_type=DogeCoinDataset,
    plugin_config=BigQueryConfig(ProjectID="flyte"),
    query_template="SELECT * FROM `bigquery-public-data.crypto_dogecoin.transactions` WHERE version = @version LIMIT 10;",
)

flyte.TaskEnvironment.from_task("bigquery_task_templatized_query_env", bigquery_task_templatized_query)

bigquery_env = flyte.TaskEnvironment(
    name="bigquery_env",
    image=flyte.Image.from_debian_base(name="bigquery").with_pip_packages("flyteplugins-connectors[bigquery]", "pandas"),
)


# DataFrame transformer can convert query result to pandas dataframe here.
# We can also change ``pandas.dataframe`` to ``pyarrow.Table``, and convert result to an Arrow table.
@bigquery_env.task
async def convert_bq_table_to_pandas_dataframe(df: DogeCoinDataset) -> pd.DataFrame:
    return await df.open(pd.DataFrame).all()


@bigquery_env.task
async def full_bigquery_wf(version: int) -> pd.DataFrame:
    df = await bigquery_task_templatized_query(version=version)
    return await convert_bq_table_to_pandas_dataframe(df=df)

# To run this task locally, you can use the following command:
#
# `flyte run --local bigquery_connector.py full_bigquery_wf --version 1`
#
# Check query result on bigquery console: `https://console.cloud.google.com/bigquery`

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(full_bigquery_wf, version=1)
    print(r.name)
    print(r.url)
    r.wait()
