# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "pandas",
#    "pyarrow",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment all}}
import pandas as pd
import flyte
import flyte.io

env = flyte.TaskEnvironment(
    name="data_etl",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
)


@env.task
async def extract() -> pd.DataFrame:
    # Read raw transaction records (stand-in for a real source).
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 3],
            "amount": [10.0, 5.0, 20.0, 7.5, 2.5, 1.0],
        }
    )


@env.task
async def transform(df: pd.DataFrame) -> flyte.io.DataFrame:
    # Clean and aggregate into a per-user feature table.
    df = df[df["amount"] > 0]
    agg = df.groupby("user_id", as_index=False)["amount"].sum()
    # StructuredDataset becomes flyte.io.DataFrame.
    return flyte.io.DataFrame.from_df(agg)


@env.task
async def load(sd: flyte.io.DataFrame) -> int:
    df = await sd.open(pd.DataFrame).all()
    return len(df)


@env.task
async def main() -> int:
    raw = await extract()
    features = await transform(raw)
    return await load(features)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
