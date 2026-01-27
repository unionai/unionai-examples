# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "requests"
# ]
# main = "main"
# params = "data=[1,2,3,4,5,6,7,8,9,10]"
# ///

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(
    "hello_world",
    image=flyte.Image.from_debian_base().with_pip_packages("requests"),
)

@env.task
def mean(data: list[float]) -> float:
    return sum(data) / len(data)

@env.task
def main(data: list[float]) -> float:
    output = mean(data)

    # ✅ performing trivial operations in a workflow is allowed
    output = output / 100

    # ✅ if/else is allowed
    if output < 0:
        raise ValueError("Output cannot be negative")

    return output
# {{/docs-fragment all}}

if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, data=[float(x) for x in range(10)])
    print(r.name)
    print(r.url)
    r.wait()
