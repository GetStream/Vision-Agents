FROM python:3.12-slim

# Pick up Debian security patches that aren't baked into the base tag yet.
RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12
# Keep the container's venv/cache off the bind-mounted host filesystem,
# otherwise running this on macOS overwrites the host's .venv with linux binaries.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv-cache

# Run as a non-root user. uv writes the project venv to UV_PROJECT_ENVIRONMENT
# and the cache to UV_CACHE_DIR, both pre-created and chowned here so the
# `uv sync` at container start can populate them.
RUN useradd --create-home --shell /bin/sh appuser \
    && mkdir -p /opt/venv /opt/uv-cache \
    && chown -R appuser:appuser /opt/venv /opt/uv-cache

USER appuser

WORKDIR /app

CMD ["uv", "run", "python", "-c", "import liteav; print('liteav OK')"]
