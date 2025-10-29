# # BigQuery connector example usage
#
# {{run-on-union}}
#
# This example shows how to use a Flyte BigQueryTask to execute a query.

import flyte
import pandas as pd
from flyte.io import DataFrame
from flyteplugins.connectors.bigquery.task import BigQueryConfig, BigQueryTask
from typing_extensions import Annotated

# This is the world's simplest query. Note that in order for registration to work properly, you'll need to give your
# BigQuery task a name that's unique across your project/domain for your Flyte installation.
bigquery_task_no_io = BigQueryTask(
    name="sql.bigquery.no_io",
    inputs={},
    query_template="SELECT 1",
    task_config=BigQueryConfig(ProjectID="flyte"),
)


bigquery_env_no_io_env = flyte.TaskEnvironment.from_task("bigquery_env_no_io_env", bigquery_task_no_io)


# Of course, in real world applications we are usually more interested in using BigQuery to query a dataset.
# In this case we use crypto_dogecoin data which is public dataset in BigQuery.
# [here](https://console.cloud.google.com/bigquery?project=bigquery-public-data&page=table&d=crypto_dogecoin&p=bigquery-public-data&t=transactions)
#
# Let's look out how we can parameterize our query to filter results for a specific transaction version, provided as a user input
# specifying a version.

DogeCoinDataset = Annotated[DataFrame, {"hash": str, "size": int, "block_number": int}]

bigquery_task_templatized_query = BigQueryTask(
    name="sql.bigquery.w_io",
    # Define inputs as well as their types that can be used to customize the query.
    inputs={"version": int},
    output_structured_dataset_type=DogeCoinDataset,
    task_config=BigQueryConfig(ProjectID="flyte"),
    query_template="SELECT * FROM `bigquery-public-data.crypto_dogecoin.transactions` WHERE version = @version LIMIT 10;",
)


bigquery_env = flyte.TaskEnvironment(
    name="bigquery_env",
    image=flyte.Image.from_debian_base(name="bigquery").with_pip_packages("flyteplugins-connectors[bigquery]"),
    depends_on=[bigquery_env_no_io_env],
)



# StructuredDataset transformer can convert query result to pandas dataframe here.
# We can also change `pandas.dataframe`` to `pyarrow.Table``, and convert result to an Arrow table.
@bigquery_env.task
def convert_bq_table_to_pandas_dataframe(sd: DogeCoinDataset) -> pd.DataFrame:
    return sd.open(pd.DataFrame).all()


@bigquery_env.task
def full_bigquery_wf(version: int) -> pd.DataFrame:
    sd = bigquery_task_templatized_query(version=version)
    return convert_bq_table_to_pandas_dataframe(sd=sd)


# Check query result on bigquery console: `https://console.cloud.google.com/bigquery`
#
