# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0",
#    "mcp",
#    "starlette",
#    "uvicorn",
# ]
# ///

"""A basic MCP server app that serves a custom FastMCP instance.

This example shows how to deploy any FastMCP server as a Flyte app using
``MCPAppEnvironment``. The server exposes tools via the Model Context Protocol
(MCP) over HTTP.
"""

import flyte
from flyte.ai.mcp import MCPAppEnvironment
from mcp.server.fastmcp import FastMCP

# {{docs-fragment basic-mcp}}
mcp = FastMCP(name="demo-generic-mcp")


@mcp.tool()
def ping() -> str:
    """Health-style echo for demos."""
    return "pong"


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


env = MCPAppEnvironment(
    name="generic-mcp-demo",
    mcp=mcp,
    transport="streamable-http",
    mcp_mount_path="/mcp",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


if __name__ == "__main__":
    flyte.init_from_config()
    handle = flyte.serve(env)
    handle.activate(wait=True)
    print(f"App is ready at {handle.endpoint}")
# {{/docs-fragment basic-mcp}}
