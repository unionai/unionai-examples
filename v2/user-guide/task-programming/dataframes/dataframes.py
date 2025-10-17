# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
#    "pandas",
#    "pyarrow",
#    "numpy"
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
from typing import Annotated

import numpy as np
import pandas as pd
import flyte
import flyte.io

env = flyte.TaskEnvironment(
    "dataframe_usage",
    image= flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow", "numpy"),
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

BASIC_EMPLOYEE_DATA = {
    "employee_id": range(1001, 1009),
    "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah"],
    "department": ["HR", "Engineering", "Engineering", "Marketing", "Finance", "Finance", "HR", "Engineering"],
    "hire_date": pd.to_datetime(
        ["2018-01-15", "2019-03-22", "2020-07-10", "2017-11-01", "2021-06-05", "2018-09-13", "2022-01-07", "2020-12-30"]
    ),
}

ADDL_EMPLOYEE_DATA = {
    "employee_id": range(1001, 1009),
    "salary": [55000, 75000, 72000, 50000, 68000, 70000, np.nan, 80000],
    "bonus_pct": [0.05, 0.10, 0.07, 0.04, np.nan, 0.08, 0.03, 0.09],
    "full_time": [True, True, True, False, True, True, False, True],
    "projects": [
        ["Recruiting", "Onboarding"],
        ["Platform", "API"],
        ["API", "Data Pipeline"],
        ["SEO", "Ads"],
        ["Budget", "Forecasting"],
        ["Auditing"],
        [],
        ["Platform", "Security", "Data Pipeline"],
    ],
}
# {{/docs-fragment setup}}


# {{docs-fragment raw-dataframe}}
@env.task
async def create_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(BASIC_EMPLOYEE_DATA)
# {{docs-fragment raw-dataframe}}


# {{docs-fragment from-df}}
@env.task
async def create_flyte_dataframe() -> Annotated[flyte.io.DataFrame, "parquet"]:
    pd_df = pd.DataFrame(ADDL_EMPLOYEE_DATA)
    fdf = flyte.io.DataFrame.from_df(pd_df)
    return fdf
# {{/docs-fragment from-df}}


# {{docs-fragment automatic}}
@env.task
async def join_data(raw_dataframe: pd.DataFrame, flyte_dataframe: pd.DataFrame) -> flyte.io.DataFrame:
    joined_df = raw_dataframe.merge(flyte_dataframe, on="employee_id", how="inner")
    return joined_df
# {{/docs-fragment automatic}}


# {{docs-fragment download}}
@env.task
async def download_data(joined_df: flyte.io.DataFrame):
    downloaded = await joined_df.open(pd.DataFrame).all()
    print("Downloaded Data:\n", downloaded)
# {{/docs-fragment download}}


# {{docs-fragment main}}
@env.task
async def main():
    raw_df = await create_raw_dataframe ()
    flyte_df = await create_flyte_dataframe ()
    joined_df = await join_data (raw_df, flyte_df)
    await download_data (joined_df)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment main}}