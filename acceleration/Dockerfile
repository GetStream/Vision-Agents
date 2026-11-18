# syntax=docker/dockerfile:1
# The router, for local development.
#
# This image is what compose.yaml builds and nothing else. Production does not
# use it: the release binaries are published to S3 by CI, and the runtime image
# that fetches and runs them is built in the chat repository, which is where the
# knowledge of how to deploy onto our clusters lives. What is here is the same
# binary in the simplest possible wrapper, so that `docker compose up` works
# without an S3 bucket or a version registry.
#
# There is no proxy here either. On Stream's deployment the router sits behind
# stream-accelerate, which authenticates Stream API keys against Stream's own app
# records and lives in the chat repository for that reason. Running this service
# on its own means running the router in api_key mode instead, where it
# authenticates callers itself.
FROM golang:1.27-bookworm AS build

WORKDIR /src

COPY go.mod go.sum ./
# goaec3, the echo canceller, is a private GetStream module. Pass a read-only
# token as a build secret (compose does, from GOAEC3_READ_TOKEN); it is used for
# this one download and never written into a layer.
ENV GOPRIVATE=github.com/GetStream/goaec3
RUN --mount=type=secret,id=goaec3_token,required=true \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="url.https://x-access-token:$(cat /run/secrets/goaec3_token)@github.com/GetStream/goaec3.insteadOf" \
    GIT_CONFIG_VALUE_0="https://github.com/GetStream/goaec3" \
    go mod download

COPY . .
# Go's SIMD experiment compiles the Opus codec's vector kernels, on amd64 and arm64.
ENV GOEXPERIMENT=simd
RUN CGO_ENABLED=0 go build -trimpath -o /out/router ./cmd/router \
    && CGO_ENABLED=0 go build -trimpath -o /out/usage ./cmd/usage

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/router /usr/local/bin/router
# The usage service runs from the same image: `--entrypoint usage`, or `command` in compose.
COPY --from=build /out/usage /usr/local/bin/usage

EXPOSE 8080 8070
ENTRYPOINT ["router"]
