---
name: deploy
description: Ship acceleration/cmd/router to Stream's hosted environments. Read before deploying the router or asking why the hosted one behaves differently.
---

# Deploy the router

The hosted router behind `accelerate.gcp.stream-io-api.com` is `acceleration/cmd/router`
built from this repo. **The deploy itself lives in Stream's private infra repo**
(`GetStream/chat`), and that is where to look for it: `infra/docs/accelerate_launch.md` is
the runbook, `infra/charts/accelerate/README.md` the chart. The operator CLI (`rocky`),
cluster and bucket names, service accounts and credentials are all there.

Nothing from that repo gets copied into this one. This is a public repository: name the
runbook, not the infrastructure.

## What ships

Kubernetes owns the image, a version registry owns the binary. A pod fetches
`router-linux-<arch>` at start from the registry its chart names, so shipping a router is
publishing a binary, moving that pointer and restarting — not an image build and not a
chart deploy. `cmd/agent`, the examples and the Python SDK are not part of it.

## Build

The router is cgo (opus, opusfile, soxr), so each architecture builds on its own
architecture rather than cross-compiling, and both `amd64` and `arm64` must be published or
the deploy rejects the version. A workflow in the private repo builds them from the tag
`accelerate-vX.Y.Z` on this repo, so tag and push here first.

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
`getstream-go-webrtc` module gets in without handing a token to a container. Repeat with
`--platform linux/amd64`, which works under emulation.

A version is `vX.Y.Z` or `vX.Y.Z-<prerelease>`. An unreleased build is named after the
commit it came from — `v0.6.9-dev-<short sha>` — and a published version is immutable, so
ship a new one rather than replacing one. The build is reproducible: the same commit
through the same image gives the same sha256, which is what lets the checksum a pod
reports be traced back to a commit.

## Before rolling pods

- Commit and push first, and build from a clean tree: the version has to name a commit somebody else can check out.
- Read the chart's environment against the branch you are shipping. Values and router are one deployment: a rollout of the auth work also had to move `ROUTER_AUTH_MODE` to `proxy`, because `noauth` had changed meaning underneath it. Render the chart, diff it against the live release, and expect to explain every line.
- The router runs its migrations on start. Additive is fine; anything else wants a plan before the pods roll.

## After

- The fetch initContainer's log is the only place a running pod says which binary it has. The version is deliberately not in the pod spec.
- A model whose key is missing in that environment drops to the next candidate, and when a strict term like `diarize` leaves no candidate the STT socket simply closes on the caller. Check the environment has a key for the config's head model before reading it as a bug.
- Verify from outside, not from the cluster: `examples/routers/stt_realtime_example -staging` streams a file through the hosted router and prints which model answered.
