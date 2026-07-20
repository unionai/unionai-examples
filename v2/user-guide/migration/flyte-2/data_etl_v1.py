import pandas as pd
from flytekit import task, workflow
from flytekit.types.structured import StructuredDataset


@task
def extract() -> pd.DataFrame:
    # Read raw transaction records (stand-in for a real source).
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 3],
            "amount": [10.0, 5.0, 20.0, 7.5, 2.5, 1.0],
        }
    )


@task
def transform(df: pd.DataFrame) -> StructuredDataset:
    # Clean and aggregate into a per-user feature table.
    df = df[df["amount"] > 0]
    agg = df.groupby("user_id", as_index=False)["amount"].sum()
    return StructuredDataset(dataframe=agg)


@task
def load(sd: StructuredDataset) -> int:
    df = sd.open(pd.DataFrame).all()
    return len(df)


@workflow
def main() -> int:
    raw = extract()
    features = transform(df=raw)
    return load(sd=features)
