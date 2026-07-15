# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-polars",
#    "polars",
# ]
# main = "main"
# params = ""
# ///

# {{docs-fragment setup}}
import flyte

image = flyte.Image.from_debian_base(name="polars").with_pip_packages("flyteplugins-polars")

env = flyte.TaskEnvironment(
    name="polars_env",
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)
# {{/docs-fragment setup}}


# {{docs-fragment dataframe}}
import polars as pl


@env.task
def make_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "category": ["A", "B", "A"],
            "salary": [55000.0, 75000.0, 72000.0],
            "active": [True, False, True],
        }
    )


@env.task
def summarize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("active"))
        .group_by("category")
        .agg(pl.col("salary").mean().alias("avg_salary"), pl.len().alias("count"))
        .sort("category")
    )


@env.task
def main() -> pl.DataFrame:
    return summarize(make_dataframe())
# {{/docs-fragment dataframe}}


# {{docs-fragment lazyframe}}
@env.task
def lazy_summary(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.filter(pl.col("active"))
        .group_by("category")
        .agg(pl.col("salary").mean().alias("avg_salary"))
        .sort("category")
    )
# {{/docs-fragment lazyframe}}


# {{docs-fragment interop}}
import flyte.io


@env.task
def to_flyte_df(df: pl.DataFrame) -> flyte.io.DataFrame:
    return flyte.io.DataFrame.wrap_df(df)


@env.task
def from_flyte_df(df: flyte.io.DataFrame) -> pl.DataFrame:
    return df  # returned to the caller as a Polars DataFrame
# {{/docs-fragment interop}}


# {{docs-fragment run}}
if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
# {{/docs-fragment run}}
