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
