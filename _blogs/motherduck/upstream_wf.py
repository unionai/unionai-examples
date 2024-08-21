from typing import Annotated

import pandas as pd
from flytekit import task, workflow, ImageSpec

from duckdb_artifacts import RecentEcommerceData

image = ImageSpec(
    name="test-image",
    # registry=os.environ.get("DOCKER_REGISTRY", None),
    registry="ghcr.io/dansola",
    apt_packages=["git"],
    packages=["pandas==2.2.2", "pyarrow==16.1.0"],
)

@task(container_image=image)
def get_pandas_df() -> Annotated[pd.DataFrame, RecentEcommerceData]:
    df = pd.read_csv('/root/Year 2010-2011-Table 1.csv')
    df['dt'] = pd.to_datetime(df['InvoiceDate'])

    # Find the oldest date in the dataset
    oldest_date = df['dt'].min()

    # Filter for data from the oldest month
    start_of_oldest_month = oldest_date.replace(day=1)
    end_of_oldest_month = (start_of_oldest_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
    oldest_month_data = df[(df['dt'] >= start_of_oldest_month) & (df['dt'] <= end_of_oldest_month)]

    return RecentEcommerceData.create_from(oldest_month_data)


@workflow
def wf() -> pd.DataFrame:
    return get_pandas_df()


