# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0",
#    "mcp",
#    "starlette",
#    "uvicorn",
# ]
# ///

"""A Flyte MCP server with filtered tools, scripting, and search.

This example shows how to deploy a more restricted MCP server that exposes
task, run, script, and search tools, with a task allowlist to restrict which
tasks can be accessed and configurable search paths for documentation.
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment

# Create an image with the required dependencies and search corpora baked in
image = (
    flyte.Image.from_debian_base()
    .with_apt_packages("ca-certificates", "git", "curl")
    .with_pip_packages("mcp", "starlette", "uvicorn")
)

# {{docs-fragment flyte-mcp-filtered}}
mcp_env = FlyteMCPAppEnvironment(
    name="restricted-mcp",
    image=image,
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    transport="streamable-http",
    tool_groups=["task", "run", "script", "search"],
    task_allowlist=["my-project/my-task", "another-task"],
    # Search paths (see docstring): you need to clone/fetch these during image build.
    sdk_examples_path="/root/flyte-sdk/examples",
    docs_examples_path="/root/unionai-examples/v2",
    full_docs_path="/root/llms.txt",
    instructions=(
        "This MCP server provides tools to run and monitor specific Flyte tasks, "
        "build and run UV scripts remotely, and search Flyte SDK/docs examples. "
        "Only allowlisted tasks can be accessed."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app_handle = flyte.serve(mcp_env)
    app_handle.activate(wait=True)
    print(f"App is ready at {app_handle.endpoint}")
# {{/docs-fragment flyte-mcp-filtered}}
