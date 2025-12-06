FROM ghcr.io/astral-sh/uv-python:3.13-slim

# Set working directory
WORKDIR /app

# Copy the entire project (needed for local path dependencies)
COPY agents-core/ ./agents-core/
COPY plugins/ ./plugins/
COPY examples/01_simple_agent_example/ ./examples/01_simple_agent_example/

# Set working directory to the example
WORKDIR /app/examples/01_simple_agent_example

# Install dependencies using uv
RUN uv sync --frozen

# Default command
CMD ["uv", "run", "python", "simple_agent_example.py"]

