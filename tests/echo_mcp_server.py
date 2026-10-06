"""A stdio MCP server with one tool, for tests that need a real one to talk to."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Say the text back."""
    return text


if __name__ == "__main__":
    mcp.run()
