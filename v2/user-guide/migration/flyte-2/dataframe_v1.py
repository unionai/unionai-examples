import pandas as pd
from flytekit import task, workflow
from flytekit.types.structured import StructuredDataset


@task
def make_df() -> StructuredDataset:
    df = pd.DataFrame({"employee_id": [1, 2, 3], "salary": [50000, 60000, 70000]})
    return StructuredDataset(dataframe=df)


@task
def total_payroll(sd: StructuredDataset) -> float:
    df = sd.open(pd.DataFrame).all()
    return float(df["salary"].sum())


@workflow
def main() -> float:
    return total_payroll(sd=make_df())
