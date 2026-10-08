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
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/router ./cmd/router

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/router /usr/local/bin/router

EXPOSE 8080
ENTRYPOINT ["router"]
