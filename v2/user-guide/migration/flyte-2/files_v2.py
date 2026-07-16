# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b52",
# ]
# main = "main"
# params = "content=hello"
# ///

# {{docs-fragment all}}
import flyte
from flyte.io import File

env = flyte.TaskEnvironment(name="files")


@env.task
async def write_file(content: str) -> File:
    with open("out.txt", "w") as f:
        f.write(content)
    # File.from_local uploads the file to blob storage and returns a reference
    # (a lightweight pointer, not the materialized bytes).
    return await File.from_local("out.txt")


@env.task
async def read_file(f: File) -> str:
    async with f.open("rb") as fh:
        return (await fh.read()).decode("utf-8")


@env.task
async def main(content: str) -> str:
    f = await write_file(content)
    return await read_file(f)
# {{/docs-fragment all}}


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main, content="hello")
    print(r.name)
    print(r.url)
    r.wait()
