FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12
# Keep the container's venv/cache off the bind-mounted host filesystem,
# otherwise running this on macOS overwrites the host's .venv with linux binaries.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv-cache

WORKDIR /app

CMD ["uv", "run", "python", "-c", "import liteav; print('liteav OK')"]
