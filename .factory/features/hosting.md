# Hosting

[Sprint 9](../sprint9.md) and [sprint 17](../sprint17.md), "Hosting for acceleration". Built,
and almost none of it is in this repository.

## Asked for

Sprint 9 wanted a Docker image for running the Go API easily, and asked where these agents
should run. Sprint 17 answered it: a chart named `accelerate`, CI that compiles the Go binary
into an image the way chat does, one pod in us-east4 enabled at the region rather than the
shard, and the S3-plus-shiply deploy flow prototyped on a chat branch rather than a plain
image deploy.

## Where it lives, and why

Vision-Agents is public and is deliberately granted no access to Artifact Registry, to S3 or
to any Stream cloud account: federating a public repository into the project would put every
workflow, pull request and transitive Action inside the trust boundary. So the split is a
security boundary, not a preference. This repository produces a tag, `accelerate-vX.Y.Z`,
and the chat repository does everything after it — `build-accelerate.yml` checks out that
tag, builds `cmd/router` natively on amd64 and arm64 because it is cgo, and publishes the
binaries and their checksums to S3, where `shiply deploy -s accelerate` moves a version
registry pointer at one of them.

The runtime image carries no router at all. It is built in chat from
`projects/accelerate-fetch`, the fetcher that resolves that pointer at pod start, plus the
Opus and soxr shared libraries the binary links against. The chart, the Terraform for
Postgres and Valkey, and `stream-accelerate` — the proxy that authenticates a Stream API key
against chat's own tables, which is why this service can default to `noauth` behind it — are
all there too.

What is here is [Dockerfile](../../acceleration/Dockerfile), which is for
`docker compose up` and says in its own comments that production does not use it.

## Not done

- **Nothing is deployed.** The chart and the us-east4 shard file are on a branch, and the
  shard values still hold a `REPLACE` for the Valkey endpoint.
- **`publicURL` is empty,** so nothing external resolves the router. Telephony vendors fetch
  a call plan from `ROUTER_PUBLIC_URL` when they answer, which means inbound calls stay
  unavailable until it is a real hostname.
- **One pod, one region, and nowhere to fail over to.** It holds live conversations, so a
  restart is a dropped call rather than a retried request.
