FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir setuptools

COPY tencent_rtc_sdk /opt/tencent_rtc_sdk

WORKDIR /opt/tencent_rtc_sdk/python/python_x86_64

RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        cp /opt/tencent_rtc_sdk/lib/arm64/libliteav.so ./lib/libliteav.so; \
    else \
        cp /opt/tencent_rtc_sdk/lib/x86_64/libliteav.so ./lib/libliteav.so; \
    fi

RUN python setup.py build_ext --inplace

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir -p /opt/liteav/lib /opt/liteav/ext
COPY --from=builder /opt/tencent_rtc_sdk/python/python_x86_64/lib/libliteav.so /opt/liteav/lib/
COPY --from=builder /opt/tencent_rtc_sdk/python/python_x86_64/_liteav*.so /opt/liteav/ext/

COPY vision-agents/plugins/tencent/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV LD_LIBRARY_PATH=/opt/liteav/lib
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=3.12

WORKDIR /app

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uv", "run", "python", "-c", "import liteav; print('liteav OK')"]
