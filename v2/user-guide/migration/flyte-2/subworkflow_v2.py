# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "raw_value=42.0"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="subworkflow")


@env.task
def impute(value: float) -> float:
    # Replace missing/negative sentinel values with 0.
    return value if value >= 0 else 0.0


@env.task
def scale(value: float) -> float:
    return value / 100.0


# A preprocessing "subworkflow" is just a task that calls other tasks.
@env.task
def preprocess(value: float) -> float:
    imputed = impute(value)
    return scale(imputed)


@env.task
def main(raw_value: float) -> float:
    return preprocess(raw_value)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, raw_value=42.0)
    print(r.name)
    print(r.url)
    r.wait()
