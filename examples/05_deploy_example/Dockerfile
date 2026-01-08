FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy only the example (uses PyPI packages, not local paths)
COPY pyproject.toml uv.lock deploy_example.py ./

# Expose port if needed for health checks or callbacks
EXPOSE 8080

# Install deps at startup (faster push, slower cold start)
CMD ["sh", "-c", "uv sync --frozen && uv run deploy_example.py"]

