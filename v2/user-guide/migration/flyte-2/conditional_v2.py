# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "n_rows=50000"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="conditional")


@env.task
def train_gradient_boosting(n_rows: int) -> str:
    return f"trained gradient boosting on {n_rows} rows"


@env.task
def train_logistic_regression(n_rows: int) -> str:
    return f"trained logistic regression on {n_rows} rows"


# Branching is now ordinary Python control flow -- no conditional() DSL.
@env.task
def main(n_rows: int) -> str:
    if n_rows > 10_000:
        return train_gradient_boosting(n_rows)
    return train_logistic_regression(n_rows)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, n_rows=50000)
    print(r.name)
    print(r.url)
    r.wait()
