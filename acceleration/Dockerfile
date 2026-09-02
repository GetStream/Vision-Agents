# Two images from one file.
#
# The default is the runtime shell that runs on Kubernetes, and it carries no router. The
# binary arrives at pod start from S3, fetched by cmd/fetchbinary into an emptyDir the
# router then execs from, because the version registry rather than the pod spec is what
# says which release is running. What the shell carries is everything that does not change
# release to release: the Opus and soxr shared libraries the router links against through
# LiveKit's media-sdk, and the fetcher itself.
#
# That split is why this cannot be distroless the way the other Go services here are. The
# router is cgo, so at runtime it needs a libc and three shared objects, and a static base
# has none of them.
#
# The `dev` target is the whole thing in one image, router included, which is what
# compose.yaml builds. There is no S3 and no registry pointer locally, so the binary has to
# come from the same place it always did.

FROM --platform=$BUILDPLATFORM golang:1.26-bookworm AS build-fetcher

ARG TARGETARCH

WORKDIR /src

COPY go.mod go.sum ./
COPY cmd/fetchbinary ./cmd/fetchbinary
COPY cmd/gateway ./cmd/gateway
COPY internal/auth ./internal/auth
COPY internal/store ./internal/store
COPY migrations ./migrations

# Both are pure Go, so they cross-compile from whichever architecture the builder runs on
# rather than through emulation, and neither reaches the private getstream-go-webrtc module
# that the router does -- which is why only the directories they need are copied.
RUN CGO_ENABLED=0 GOOS=linux GOARCH="${TARGETARCH}" \
    go build -trimpath -ldflags="-s -w" -o /out/fetchbinary ./cmd/fetchbinary \
 && CGO_ENABLED=0 GOOS=linux GOARCH="${TARGETARCH}" \
    go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libopus0 libopusfile0 libsoxr0 \
    && rm -rf /var/lib/apt/lists/*

# 65532 is the uid distroless uses, so a securityContext pinning runAsUser matches this
# image and every other one in the fleet without a special case.
RUN groupadd --gid 65532 nonroot \
    && useradd --uid 65532 --gid 65532 --no-create-home --shell /usr/sbin/nologin nonroot

COPY --from=build-fetcher /out/fetchbinary /usr/local/bin/fetchbinary
COPY --from=build-fetcher /out/gateway /usr/local/bin/gateway

# Where the initContainer installs the router and where this container execs it from. It is
# a mount point: the emptyDir the chart declares is what actually lands here.
RUN mkdir -p /opt/stream/bin && chown 65532:65532 /opt/stream/bin

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/opt/stream/bin/router"]

FROM golang:1.26-bookworm AS build-router

RUN apt-get update && apt-get install -y --no-install-recommends \
      pkg-config libopus-dev libopusfile-dev libsoxr-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# getstream-go-webrtc is private, so the fetch borrows the host's SSH agent the same way a
# local `go mod download` does.
ENV GOPRIVATE=github.com/GetStream/*

COPY go.mod go.sum ./
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts \
    && git config --global url."git@github.com:".insteadOf https://github.com/ \
    && go mod download

COPY . .
RUN CGO_ENABLED=1 go build -trimpath -o /out/router ./cmd/router

FROM runtime AS dev

COPY --from=build-router /out/router /opt/stream/bin/router
