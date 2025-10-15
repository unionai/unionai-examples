# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
#    "numpy"
# ]
# main = "main"
# params = "x_list=[1,2,3,4,5,6,7,8,9,10]"
# ///

import flyte

env = flyte.TaskEnvironment(
    name="my_env",
    image=flyte.Image.from_uv_script(
            __file__,
            name="my-image",
            registry="registry.example.com/my-org" # Only needed for local builds
        )
)


@env.task
def main(x_list: list[int]) -> float:
    arr = np.array(x_list)
    return float(np.mean(arr))


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, x_list=list(range(10)))
    print(r.name)
    print(r.url)
    r.wait()
