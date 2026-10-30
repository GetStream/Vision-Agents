---
name: deploy
description: Ship acceleration/cmd/router to Stream's hosted environments. Read before deploying the router or asking why the hosted one behaves differently.
---

# Deploy the router

The hosted router behind `accelerate-<region>.gcp.stream-io-api.com` (production) and
`accelerate-staging.gcp.stream-io-api.com` is `acceleration/cmd/router`
built from this repo. **The deploy lives in Stream's private infra repo** (`GetStream/chat`):
`infra/docs/accelerate_launch.md` is the runbook, `infra/charts/accelerate/README.md` the
chart, and the CLI, cluster names and credentials are all there. This repo is public — name
the runbook, not the infrastructure.

Kubernetes owns the image, a version registry owns the binary: a pod fetches
`router-linux-<arch>` at start, so shipping a router is publishing a binary, moving that
pointer and restarting. Not an image build, not a chart deploy.

## Build

The router is cgo (opus, opusfile, soxr), so each architecture builds on its own
architecture, and both must be published or the deploy rejects the version. Tag the commit
`accelerate-vX.Y.Z` here and push the tag. The private repo's build workflow builds from that
tag, but it publishes to an S3 release bucket, not to the bucket pods fetch from; a
release reaches pods by building it by hand, then `rocky launch upload` once per architecture
and `rocky launch deploy`, which moves the pointer and restarts the pods (runbook step 8).

By hand, one container per architecture, from the repo root:

```bash
VERSION=vX.Y.Z
mkdir -p /tmp/router-release
docker run --rm --platform linux/arm64 \
  -v "$PWD":/src:ro -v "$(go env GOMODCACHE)":/go/pkg/mod -v /tmp/router-release:/out \
  -e GOWORK=off -e GOPROXY=off -e GOTOOLCHAIN=local -e VERSION="$VERSION" \
  -w /src/acceleration golang:1.27-bookworm bash -c '
    apt-get -qq update && apt-get -qq install -y --no-install-recommends \
      pkg-config libopus-dev libopusfile-dev libsoxr-dev >/dev/null
    CGO_ENABLED=1 go build -trimpath -ldflags="-s -w -X main.release=$VERSION" \
      -o /out/router-linux-arm64 ./cmd/router'
```

`GOWORK=off` because `go.work` otherwise puts the module in workspace mode, which the build
refuses; `GOPROXY=off` builds from the host module cache, so the container uses exactly the
modules the host resolved. `main.release` is the version Sentry files events under, as the
workflow sets it. Repeat with
`--platform linux/amd64`, which works under emulation.

`GOPROXY=off` also means a module the host has not downloaded fails the build: after a
`go.mod` change, run `go mod download` in `acceleration/` on the host first.

Versions are `vX.Y.Z` or `vX.Y.Z-<prerelease>`, an unreleased build is named after its
commit (`v0.6.9-dev-<short sha>`), and a published one is immutable. The build is
reproducible, so the checksum a pod reports traces back to a commit.

## Overlap deploys (staging and the benchmark release)

Staging and the benchmark release (`perfbench -target benchmark`, nobody else's traffic) ship
a router without dropping a call. The runbook is `infra/docs/accelerate_overlap_deploy.md` in
the private repo; `rocky fastdeploy --service accelerate-staging` (or `accelerate-benchmark`)
builds this checkout and swaps it in, in 11-20 s warm.

- It needs `source ~/.config/rocky/gcp.env`, Docker, Go and gcloud, and an arm64 machine:
  the build runs natively and these releases' pods run arm64 only. It builds the working tree,
  uncommitted changes included, so the clean-tree rule below is for tagged releases.
- `--status` prints what each pod runs. The fetch initContainer's log only says what a pod
  started on, which a swap has since replaced.
- On staging a fastdeploy and a tagged release move the same pointer, and a restarted pod
  comes back on whichever was last. Pull before a fastdeploy, or it rolls back a teammate's
  release. Load-test on benchmark, not staging.

- A supervisor (devrun) owns the public socket and hands it to each router as fd 3
  (`listen_fd`), with a peer port of its own (`node.listen`). A swap starts the new router on
  the same socket, waits for `/health`, then SIGTERMs the old one.
- The old router stops accepting, keeps every call until it ends, bounded by
  `shutdown.drain` (1 h), still answering its peers, then exits. A second signal ends its
  calls now. At most three generations run at once.
- A draining router exits on `session.Manager.Len()` reaching zero, which counts a session
  until its `Close` returns. Anything new that holds a call open must be counted there, or
  the old router exits under it.
- A call only ends when the client's `close` reaches the router. A client that stops reading
  its session socket, or closes the connection straight after `close`, leaves the session
  running and holds the old router for the whole drain. `Pipeline.Leave` in the Go SDK waits
  for the router to hang up.
- To check a deploy, run perfbench against the benchmark release during the swap and confirm
  every call held, nothing answered 5xx, and devrun logged `pid N (sha) drained in ...,
  exit 0` for each old router once its last call ended.

## Production

Production is three shards, us-east4, us-west1 and europe-west4. Each region has its own
hostname, `accelerate-<region>.gcp.stream-io-api.com`, that reaches only that shard, and
today only us-east4's is published (`local.accelerate_prod_regions`); it is the SDKs'
default. Any shard serves any app. Each runs the router with its own Postgres and Redis
beside the global ones in us-east4 (`acceleration/README.md`, "Shards"). The runbook is
`infra/docs/accelerate_shards_rollout.md` in the private repo.

A release is a tagged build, published once, then deployed per shard:

- `rocky launch upload` both architectures once; the binary is shared by every shard.
- `rocky launch deploy --shard <cluster> --env prod` per shard. The registry lock is global,
  so the three pointer moves run one after another; the restarts they trigger run in parallel.
- Migrations run on start and take an advisory lock, so shards starting together are safe.
  A migration has to be right for both databases: the global tables and the shard tables
  share one migration set.
- Prod runs with the overlap deploy on, so a restarted pod drains its calls (up to 1 h),
  but `rocky fastdeploy` does not reach the prod clusters yet: deploy with `launch deploy`.

## The key encryption key

A hosted router can unwrap its key encryption key with Cloud KMS at startup in place of holding
it in `ROUTER_AUTH_KEK`: the steps (IAM role, the `gcloud kms encrypt` line, the two variables)
are in `acceleration/README.md`, «The key encryption key in Cloud KMS». Ship the build first,
with the values unchanged, then switch the values and restart: in that one change set
`ROUTER_AUTH_KMS_KEY` and `ROUTER_AUTH_KEK_V1_WRAPPED` and remove `ROUTER_AUTH_KEK` (and
`ROUTER_AUTH_KEK_V1`) from the router's environment, because a router with both forms of a
version refuses to start. Keep a copy of the plain key outside the deployment for rollback,
and check that the pods start and open an existing connection. To roll back, restore
`ROUTER_AUTH_KEK` from that copy and unset `ROUTER_AUTH_KMS_KEY` and `ROUTER_AUTH_KEK_V1_WRAPPED`.

## Before and after

- Commit and push first, and build from a clean tree: the version has to name a commit somebody else can check out.
- Values and router are one deployment. Rolling the auth work also meant moving `ROUTER_AUTH_MODE` to `proxy`, because `noauth` had changed meaning underneath it. Render the chart, diff it against the live release, explain every line.
- Migrations run on start. Additive is fine; anything else wants a plan first.
- Never roll back below `accelerate-v0.6.135`, the first tag with f38926d2 (#1182): an older
  router ignores `require_approval` on connector bindings, so their write tools run without
  the person's approval. This floor holds in every environment.
- Staging has a second, higher floor: never roll back below `accelerate-v0.6.191` (#1433). Since 2026-10-29
  staging devrun holds the sandbox proxy port :8091 and passes it to the router as fd 4
  (`router.overlapDeploy.holdSandboxProxy`, GetStream/chat #19810, #19840). A router older than v0.6.191 ignores that socket and retries :8091
  forever: the API serves, the sandbox proxy does not. To go below it, in this order:
  1. Set `router.overlapDeploy.holdSandboxProxy: false`.
  2. Run `rocky accelerate up`.
  3. Wait for the rollout.
  4. Roll back.

  The steps are in GetStream/chat `infra/docs/accelerate_overlap_deploy.md`, paragraph "Revert" under
  "The sandbox proxy socket". If the only target is older than a floor and its steps do not apply, stop and ask.
- The version is deliberately not in the pod spec. The fetch initContainer's log says which binary a pod started on; after a fastdeploy, ask `--status` instead.
- A model with no key in that environment drops to the next candidate, and when a strict term like `diarize` leaves none the STT socket just closes on the caller. Check the key before reading that as a bug.
- Verify from outside: `examples/routers/stt_realtime_example -hosted` streams a file through the hosted router and prints which model answered.
