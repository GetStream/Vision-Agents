from fastmcp import FastMCP
import httpx

mcp = FastMCP("Stream MCP Server")


@mcp.tool()
def get_forecast(city: str) -> str:
    """Return today's weather for <city>."""
    data = httpx.get(f"https://wttr.in/{city}?format=%C+%t").text
    return data.strip()


if __name__ == "__main__":
    mcp.run()  # default transport = stdio
