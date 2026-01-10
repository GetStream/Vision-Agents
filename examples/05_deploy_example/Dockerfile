FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.13 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.13 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install uv
RUN pip install uv --break-system-packages

# Copy only the example (uses PyPI packages, not local paths)
COPY pyproject.toml uv.lock deploy_example.py ./

# Expose port if needed for health checks or callbacks
EXPOSE 8080

# Install deps at startup (faster push, slower cold start)
CMD ["sh", "-c", "uv sync --frozen && uv run deploy_example.py"]
