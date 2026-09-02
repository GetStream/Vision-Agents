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
#
# The router links Opus and soxr through LiveKit's media-sdk, so both stages need
# the C libraries: headers to build, shared objects to run.
FROM golang:1.26-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config libopus-dev libopusfile-dev libsoxr-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# getstream-go-webrtc is private, so the fetch borrows the host's SSH agent the same
# way a local `go mod download` does.
ENV GOPRIVATE=github.com/GetStream/*

COPY go.mod go.sum ./
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts \
    && git config --global url."git@github.com:".insteadOf https://github.com/ \
    && go mod download

COPY . .
RUN CGO_ENABLED=1 go build -trimpath -o /out/router ./cmd/router

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libopus0 libopusfile0 libsoxr0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/router /usr/local/bin/router

EXPOSE 8080
ENTRYPOINT ["router"]
