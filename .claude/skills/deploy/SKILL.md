---
name: deploy
description: Ship acceleration/cmd/router to Stream's hosted environments. Read before deploying the router or asking why the hosted one behaves differently.
---

# Deploy the router

The hosted router behind `accelerate.gcp.stream-io-api.com` is `acceleration/cmd/router`
built from this repo. **The deploy lives in Stream's private infra repo** (`GetStream/chat`):
`infra/docs/accelerate_launch.md` is the runbook, `infra/charts/accelerate/README.md` the
chart, and the CLI, cluster names and credentials are all there. This repo is public — name
the runbook, not the infrastructure.

Kubernetes owns the image, a version registry owns the binary: a pod fetches
`router-linux-<arch>` at start, so shipping a router is publishing a binary, moving that
pointer and restarting. Not an image build, not a chart deploy.

## Build

The router is cgo (opus, opusfile, soxr), so each architecture builds on its own
architecture, and both must be published or the deploy rejects the version. A workflow in
the private repo builds them from the tag `accelerate-vX.Y.Z` here, so tag and push first.

By hand, one container per architecture, from the repo root:

```bash
mkdir -p /tmp/router-release
docker run --rm --platform linux/arm64 \
  -v "$PWD":/src:ro -v "$(go env GOMODCACHE)":/go/pkg/mod -v /tmp/router-release:/out \
  -e GOWORK=off -e GOPROXY=off -e GOTOOLCHAIN=local \
  -w /src/acceleration golang:1.26-bookworm bash -c '
    apt-get -qq update && apt-get -qq install -y --no-install-recommends \
      pkg-config libopus-dev libopusfile-dev libsoxr-dev >/dev/null
    CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/router-linux-arm64 ./cmd/router'
```

`GOWORK=off` because `go.work` otherwise puts the module in workspace mode, which the build
refuses; `GOPROXY=off` builds from the host module cache, which is how the private
`getstream-go-webrtc` module gets in without handing a container a token. Repeat with
`--platform linux/amd64`, which works under emulation.

Versions are `vX.Y.Z` or `vX.Y.Z-<prerelease>`, an unreleased build is named after its
commit (`v0.6.9-dev-<short sha>`), and a published one is immutable. The build is
reproducible, so the checksum a pod reports traces back to a commit.

## Before and after

- Commit and push first, and build from a clean tree: the version has to name a commit somebody else can check out.
- Values and router are one deployment. Rolling the auth work also meant moving `ROUTER_AUTH_MODE` to `proxy`, because `noauth` had changed meaning underneath it. Render the chart, diff it against the live release, explain every line.
- Migrations run on start. Additive is fine; anything else wants a plan first.
- The fetch initContainer's log is the only place a pod says which binary it has — the version is deliberately not in the pod spec.
- A model with no key in that environment drops to the next candidate, and when a strict term like `diarize` leaves none the STT socket just closes on the caller. Check the key before reading that as a bug.
- Verify from outside: `examples/routers/stt_realtime_example -staging` streams a file through the hosted router and prints which model answered.
