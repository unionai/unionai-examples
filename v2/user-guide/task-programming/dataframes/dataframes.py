# {{docs-fragment from_df}}
@env.task
async def create_flyte_dataframe() -> Annotated[flyte.io.DataFrame, "csv"]:
    pd_df = pd.DataFrame(ADDL_EMPLOYEE_DATA)
    fdf = flyte.io.DataFrame.from_df(pd_df)
    return fdf
# {{/docs-fragment from_df}}

# {{docs-fragment native}}
@env.task
async def create_raw_dataframe() -> pd.DataFrame:
# {{/docs-fragment native}}

# {{docs-fragment annotated}}
from typing import Annotated
import pandas as pd
import flyte.io

def my_task() -> Annotated[flyte.io.DataFrame, "parquet"]:
	# create a pandas DataFrame and convert it to a flyte DataFrame
	df = pd.DataFrame(...)
	return flyte.io.DataFrame.from_df(df)
# {{/docs-fragment annotated}}

# {{docs-fragment download}}
# Download all data at once
downloaded = await flyte_dataframe.open(pd.DataFrame).all()
# or in synchronous contexts: downloaded = flyte_dataframe.open(pd.DataFrame).all()
# {{/docs-fragment download}}

# {{docs-fragment automatic}}
@env.task
async def get_employee_data(raw_dataframe: pd.DataFrame, flyte_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    This task takes two dataframes as input. We'll pass one raw pandas dataframe, and one flyte.io.DataFrame.
    Flyte automatically converts the flyte.io.DataFrame to a pandas DataFrame. The actual download and conversion
    happens only when we access the data (in this case, when we do the merge)."""
    joined_df = raw_dataframe.merge(flyte_dataframe, on="employee_id", how="inner")

    return joined_df
# {{/docs-fragment automatic}}