# hello.py

# Your sub-task definitions here

@env.task
def main(name: str):
     # The main task logic here

if __name__ == "__main__":
    flyte.init_from_config()
    flyte.run(main, name="Ada")
