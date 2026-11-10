# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0",
#    "mcp",
#    "starlette",
#    "uvicorn",
# ]
# ///

"""A Flyte MCP server with filtered tools and search.

This example shows how to deploy a more restricted MCP server that exposes an
explicit list of task, run, and search tools, with a task allowlist to restrict
which tasks can be accessed and configurable search paths for documentation.
It lists its tools instead of using tool groups so that it can leave out
`abort_run`, which the `run` group includes and no allowlist covers.
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
    # An explicit list instead of tool_groups=["task", "run", "search"], so that
    # abort_run (in the "run" group, not covered by any allowlist) is left out.
    tools=[
        "run_task", "get_task", "list_tasks",
        "get_run", "get_run_io", "list_runs", "wait_for_run", "rerun_run",
        "search_flyte_sdk_examples", "search_flyte_docs_examples", "search_full_docs",
    ],
    task_allowlist=["my-project/my-task", "another-task"],
    # Search paths (see docstring): you need to clone/fetch these during image build.
    sdk_examples_path="/root/flyte-sdk/examples",
    docs_examples_path="/root/unionai-examples/v2",
    full_docs_path="/root/llms.txt",
    instructions=(
        "This MCP server provides tools to run and monitor specific Flyte tasks, "
        "and search Flyte SDK/docs examples. "
        "Only allowlisted tasks can be accessed."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app_handle = flyte.serve(mcp_env)
    app_handle.activate(wait=True)
    print(f"App is ready at {app_handle.endpoint}")
# {{/docs-fragment flyte-mcp-filtered}}
