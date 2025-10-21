# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b0",
# ]
# main = "main"
# params = "name='World'"
# ///

# run_local_from_python.py

# {{docs-fragment all}}
import flyte

env = flyte.TaskEnvironment(name="hello_world")

@env.task
def main(name: str) -> str:
     return f"Hello, {name}!"

if __name__ == "__main__":
    flyte.init_from_config()
    r =  flyte.with_runcontext(mode="local").run(main, name="World")
    print(r.name)
    print(r.url)
    r.wait()
# {{/docs-fragment all}}
