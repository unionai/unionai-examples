import pathlib

from dep import foo

import flyte

env = flyte.TaskEnvironment(
    name="full_build",
    image=flyte.Image.from_debian_base().with_source_folder(
        pathlib.Path(__file__).parent,
        copy_contents_only=True  # Avoid nested folders
    ),
)


@env.task
def square(x) -> int:
    return x ** foo()


@env.task
def main(n: int) -> list[int]:
    return list(flyte.map(square, range(n)))


if __name__ == "__main__":
    # copy_contents_only=True requires root_dir=parent, False requires root_dir=parent.parent
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.with_runcontext(copy_style="none", version="x").run(main, n=10)
    print(run.url)
