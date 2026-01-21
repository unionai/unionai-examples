import marimo
import marimo as mo

__generated_with = "0.9.21"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Getting Started with Union

        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        # Step 2: Let's Log into Union


        """
    )
    return


@app.cell
def __():
    import subprocess

    # Define the command and its arguments as a list
    command = ["union", "create", "login", "--auth", "device-flow", "--host", "https://demo.hosted.unionai.cloud"]

    # Start the subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Read and display the output line by line
    for line in process.stdout:
        print(line, end='')

    # Wait for the subprocess to complete
    process.wait()

if __name__ == "__main__":
    app.run()

