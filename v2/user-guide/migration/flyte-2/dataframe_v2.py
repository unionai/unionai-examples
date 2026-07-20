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
    name="dataframe",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
)


@env.task
async def make_df() -> flyte.io.DataFrame:
    df = pd.DataFrame({"employee_id": [1, 2, 3], "salary": [50000, 60000, 70000]})
    # StructuredDataset becomes flyte.io.DataFrame.
    return flyte.io.DataFrame.from_df(df)


@env.task
async def total_payroll(fdf: flyte.io.DataFrame) -> float:
    df = await fdf.open(pd.DataFrame).all()
    return float(df["salary"].sum())


@env.task
async def main() -> float:
    return await total_payroll(await make_df())
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.name)
    print(r.url)
    r.wait()
