# hello.py

... # Your other task definitions here

@env.task
def main(name: str):
     ... # The main task logic here

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.with_runcontext(mode="local").run(main)
