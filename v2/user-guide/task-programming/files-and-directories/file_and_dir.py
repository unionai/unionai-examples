# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte==2.0.0b31",
# ]
# main = "create_and_check_dir"
# params = ""
# ///

# {{docs-fragment write-file}}
import asyncio
import tempfile
from pathlib import Path

import flyte
from flyte.io import Dir, File

env = flyte.TaskEnvironment(name="files-and-folders")


@env.task
async def write_file(name: str) -> File:

    # Create a file and write some content to it
    with open("test.txt", "w") as f:
        f.write(f"hello world {name}")

    # Upload the file using flyte
    uploaded_file_obj = await File.from_local("test.txt")
    return uploaded_file_obj

# {{/docs-fragment write-file}}

# {{docs-fragment write-and-check-files}}
@env.task
async def write_and_check_files() -> Dir:
    coros = []
    for name in ["Alice", "Bob", "Eve"]:
        coros.append(write_file(name=name))

    vals = await asyncio.gather(*coros)
    temp_dir = tempfile.mkdtemp()
    for file in vals:
        async with file.open("rb") as fh:
            contents = await fh.read()
            # Convert bytes to string
            contents_str = contents.decode('utf-8') if isinstance(contents, bytes) else str(contents)
            print(f"File {file.path} contents: {contents_str}")
            new_file = Path(temp_dir) / file.name
            with open(new_file, "w") as out:  # noqa: ASYNC230
                out.write(contents_str)
    print(f"Files written to {temp_dir}")

    # walk the directory and ls
    for path in Path(temp_dir).iterdir():
        print(f"File: {path.name}")

    my_dir = await Dir.from_local(temp_dir)
    return my_dir

# {{/docs-fragment write-and-check-files}}

# {{docs-fragment create-and-check-dir}}
@env.task
async def check_dir(my_dir: Dir):
    print(f"Dir {my_dir.path} contents:")
    async for file in my_dir.walk():
        print(f"File: {file.name}")
        async with file.open("rb") as fh:
            contents = await fh.read()
            # Convert bytes to string
            contents_str = contents.decode('utf-8') if isinstance(contents, bytes) else str(contents)
            print(f"Contents: {contents_str}")


@env.task
async def create_and_check_dir():
    my_dir = await write_and_check_files()
    await check_dir(my_dir=my_dir)


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(create_and_check_dir)
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment create-and-check-dir}}
