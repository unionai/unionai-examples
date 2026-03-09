# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-polars>=2.0.0",
#    "polars",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import polars as pl

import flyte

env = flyte.TaskEnvironment(
    name="polars-dataframes",
    image=flyte.Image.from_debian_base(name="polars").with_pip_packages(
        "flyteplugins-polars>=2.0.0", "polars"
    ),
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)

EMPLOYEE_DATA = {
    "employee_id": [1001, 1002, 1003, 1004, 1005, 1006],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona"],
    "department": ["Engineering", "Engineering", "Marketing", "Finance", "Finance", "Engineering"],
    "salary": [75000, 72000, 50000, 68000, 70000, 80000],
    "years_experience": [5, 4, 2, 6, 5, 7],
}
# {{/docs-fragment setup}}


# {{docs-fragment polars-dataframe}}
@env.task
async def create_dataframe() -> pl.DataFrame:
    """Create a Polars DataFrame.

    Polars DataFrames are passed between tasks as serialized Parquet files
    stored in the Flyte blob store — no manual upload required.
    """
    return pl.DataFrame(EMPLOYEE_DATA)


@env.task
async def filter_high_earners(df: pl.DataFrame) -> pl.DataFrame:
    """Filter and enrich a Polars DataFrame."""
    return (
        df.filter(pl.col("salary") > 60000)
        .with_columns(
            (pl.col("salary") / pl.col("years_experience")).alias("salary_per_year")
        )
        .sort("salary", descending=True)
    )
# {{/docs-fragment polars-dataframe}}


# {{docs-fragment polars-lazyframe}}
@env.task
async def create_lazyframe() -> pl.LazyFrame:
    """Create a Polars LazyFrame.

    LazyFrames defer computation until collected, allowing Polars to
    optimize the full query plan. They are serialized to Parquet just
    like DataFrames when passed between tasks.
    """
    return pl.LazyFrame(EMPLOYEE_DATA)


@env.task
async def aggregate_by_department(lf: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate salary statistics by department using a LazyFrame.

    The query plan is built lazily and executed only when collect() is called.
    """
    return (
        lf.group_by("department")
        .agg(
            pl.col("salary").mean().alias("avg_salary"),
            pl.col("salary").max().alias("max_salary"),
            pl.len().alias("headcount"),
        )
        .sort("avg_salary", descending=True)
        .collect()
    )
# {{/docs-fragment polars-lazyframe}}


# {{docs-fragment main}}
@env.task
async def main():
    df = await create_dataframe()
    filtered = await filter_high_earners(df=df)
    print("High earners:")
    print(filtered)

    lf = await create_lazyframe()
    summary = await aggregate_by_department(lf=lf)
    print("Department summary:")
    print(summary)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment main}}
