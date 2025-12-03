# {{docs-fragment full-build}}
import pathlib
import flyte

env = flyte.TaskEnvironment(
    name="full_build",
    image=flyte.Image.from_debian_base().with_source_folder(
        pathlib.Path(__file__).parent, 
        copy_contents_only=True
    ),
)

@env.task
def main(n: int) -> list[int]:
    return list(range(n))

if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    
    # Disable fast deployment, force full container build
    run = flyte.with_runcontext(
        copy_style="none", 
        version="v1.0"
    ).run(main, n=10)
    
    print(run.url)
# {{/docs-fragment full-build}}

# Original implementation with dependencies
from dep import foo

env_with_deps = flyte.TaskEnvironment(
    name="full_build_with_deps",
    image=flyte.Image.from_debian_base().with_source_folder(pathlib.Path(__file__).parent, copy_contents_only=True),
)

@env_with_deps.task
def square(x) -> int:
    return x ** foo()

@env_with_deps.task
def main_with_deps(n: int) -> list[int]:
    return list(flyte.map(square, range(n)))
