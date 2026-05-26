# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0",
#    "mcp",
#    "starlette",
#    "uvicorn",
# ]
# ///

"""A Flyte MCP server app that exposes Flyte operations as MCP tools.

This example deploys an MCP (Model Context Protocol) server that allows AI
assistants and LLM-based clients to interact with the Flyte control plane
using the standardized MCP protocol.

The server exposes tools for running tasks, monitoring runs, managing apps
and triggers, building container images, building and running UV scripts
remotely, and searching Flyte SDK/docs examples.

Requirements:
    pip install 'flyte[mcp]'

Usage:

    Deploy all tools

    $ python v2/user-guide/build-mcp/flyte_mcp_app.py

    Or serve locally for development (recommended: `uvx`)

    $ uvx --from "flyte[mcp]" flyte-mcp
"""

import flyte
from flyte.ai.mcp import FlyteMCPAppEnvironment


# {{docs-fragment flyte-mcp-all-tools}}
# Deploy an MCP server with all tools enabled
mcp_env = FlyteMCPAppEnvironment(
    name="flyte-mcp-server",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    transport="streamable-http",
    instructions=(
        "This MCP server provides tools to interact with the Flyte control plane. "
        "Use the available tools to run tasks, monitor runs, manage apps, build images, "
        "build and run UV scripts remotely, and search SDK/docs examples."
    ),
)

if __name__ == "__main__":
    flyte.init_from_config()
    app_handle = flyte.serve(mcp_env)
    app_handle.activate(wait=True)
    print(f"App is ready at {app_handle.endpoint}")
# {{/docs-fragment flyte-mcp-all-tools}}
