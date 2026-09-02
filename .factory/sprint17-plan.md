# Sprint 17 implementation plan

Hosting `acceleration/` on GKE in `us-east4` as one pod, on the same chart and deploy
machinery chat uses. Six workstreams across three repos: `Vision-Agents` (the service code),
`chat` (the release build, the chart and the region toggle), `stream-infra` (the shiply
descriptor). Workstreams 2 and 5 gate everything else and should go first.

One constraint shapes the whole release path: **no new identity federation anywhere.** The
public repo is granted no access to any Stream cloud account, and the pod holds no federated
credential. What that costs and where it is not absolutely achievable is set out below.

---

## Findings that shape the plan

**One pod is the architecture, not a stepping stone.** The scope fence in `sprint17.md`
matches the code rather than merely being cautious. `internal/session/manager.go:86` holds
live sessions in a map, `internal/agent/streamedge/` holds WebRTC peer connections
in-process, and `internal/dispatch/dispatch.go` holds the worker pool in RAM — so an
inbound webhook landing on instance A cannot reach a worker connected to instance B.
Worse, `cmd/router/main.go` calls `simulations.Abandon(ctx)` at boot, and
`internal/store/simulations.go:316` scopes only on `started_at`, erroring *every* `simulation_runs` row in state `running`,
not just this instance's. A second replica would fail inbound calls intermittently and
mark the first one's in-flight runs as crashed. Horizontal scale is a later sprint.

**Federating a public repo into the cloud project is the wrong trade.** The obvious path —
bind a Workload Identity provider to `GetStream/Vision-Agents` so its Actions can push to
Artifact Registry, as chat does for itself — hands every workflow in an open-source repo the
ability to mint credentials in Stream's GCP project. Pull requests, workflow edits and
transitively any compromised Action in that repo all sit inside that trust boundary, and the
blast radius is a production project rather than a repo. The build therefore runs in `chat`,
which is already trusted, and reads the acceleration source at a tag. The public repo gets
no credential at all, which is a stronger property than any scoping of one would give.

The same reasoning removes the pod's GCP OIDC → AWS STS exchange and the KSA→GSA binding
that the chat branch uses. The pod reads S3 with a static, tightly scoped IAM key delivered
through Secret Manager instead. This is a real trade, not a free win: a static key must be
owned and rotated, where a federated one expires by itself. It is recorded as such in the
prerequisites rather than presented as strictly better.

**The runtime image cannot be distroless.** Every other Go service at Stream lands on
`gcr.io/distroless/static-debian12:nonroot` with `CGO_ENABLED=0`. The router links Opus and
soxr through LiveKit's media-sdk, so `acceleration/Dockerfile` already sets `CGO_ENABLED=1`
and installs `libopus0 libopusfile0 libsoxr0` in its runtime stage. The base stays
`debian:bookworm-slim`, pinned by digest, with a `nonroot` user at UID 65532 to match the
numeric id the rest of the fleet uses.

**Which is precisely why the S3 flow fits.** In the chat branch the image demotes to a
slow-moving runtime shell and the binary arrives from S3 at pod start. Here the shell is
the thing carrying the `.so` files, so image and binary have genuinely different cadences:
the shared-library set changes almost never, the router changes constantly. The two halves
of `sprint17.md` — "compile the Go binary into an image" and "use the s3 + shiply deploy
flow" — are both satisfied, one per artifact.

**Both architectures must ship, and CGO makes cross-compiling painful.** `internal/gke/artifact.go`
in shiply does a `HeadObject` on the binary *and* its `.sha256` for both amd64 and arm64
before it will deploy, because the node pools are mixed (n4/n4d x86, c4a/n4a Axion arm64)
and a missing arch surfaces only as a crashloop once a pod lands on that node.
Cross-compiling CGO needs a cross toolchain plus arm64 opus/soxr. Build each arch natively
instead — `ubuntu-24.04` and `ubuntu-24.04-arm` in a matrix — and the problem disappears.

**`/health` is the wrong liveness probe.** It returns **503** when Postgres or Redis are
configured but failing (a store that is simply unconfigured still returns 200). As a
liveness probe that converts a database blip into a pod kill. Readiness gets `/health`;
liveness gets a `tcpSocket` on 8080. No code change.

**The Valkey client supports neither auth nor TLS.** `internal/live/live.go` builds
`rueidis.ClientOption{InitAddress: []string{addr}}` from `ROUTER_REDIS_ADDR`, a bare
`host:port`. Memorystore for Valkey with AUTH or in-transit encryption enabled will not
connect. Since the sprint asks for practices "similar to the chat api", add the two options
rather than provisioning the instance without them.

**The shiply note in `sprint17.md` is inverted.** `stream-infra` is checked out on
`thierry/shiply-gke-deploy`, which is 10 commits *ahead* of the last-fetched `origin/main`
and 0 behind — those 10 commits are the GKE deploy feature itself. The risk is that the
work is unmerged, not that the checkout is stale. Fetch before starting anyway.

**Shiply may refuse a service outside the chat repo.** `ValidateChatDir` was patched on that
branch to "accept the chat monorepo layout", and `internal/aws/aws.go` hardcodes
`S3ReleasePrefix = "releases/r/GetStream/chat"`. Per-service `artifact.s3_path` overrides
the prefix, and GKE targets reject `--build` outright, so no source checkout should be
needed — but this gates the entire deploy path and must be confirmed first.

**Auth has a research document already.** `.factory/features/auth.md` covers key format,
storage, revocation, caching and the two awkward callers. `sprint17.md` asks for JWT rather
than the HMAC scheme that document explores; that is the Stream model, and the document's
own "The dashboard is a browser" section describes it — the key names the app, the secret
stays server-side and signs a JWT, the client only ever holds key and token. Its "What to
build first" list is the order to follow.

---

## Workstream 1 — Auth modes in the router

A mode switch, `ROUTER_AUTH_MODE`, with two values.

**`noauth`** is what the Stream deploy runs. The principal comes from headers the proxy
sets: `X-Stream-Organization-Id` and `X-Stream-App-Id`. The middleware must **strip any
client-supplied copy of those headers before reading them**; otherwise the mode is
spoofable by anyone who can reach the pod. This is why the NetworkPolicy in Workstream 4
is load-bearing rather than decorative.

**`api_key`** is what open source runs. Keys belong to apps, apps belong to organizations.
Verify the key, verify the JWT signed with the app secret, and resolve the principal from
the key rather than trusting the caller.

Keep the blast radius small. Roughly twenty tables key on `customer_id`; do **not** rename
the column. Resolve a principal that yields the existing `customer_id` from the app id, and
carry `organization_id` only where it is genuinely needed — rate limiting and the three new
tables. `server.go:33` and `:196` document the current behaviour honestly and both comments
should be replaced, not left describing a state that no longer holds.

Leave the three deliberately unauthenticated endpoints alone: `/health`,
`/v1/phone/hooks/stream` (HMAC-verified against `STREAM_API_SECRET` in `callhooks.go:20`)
and `/v1/phone/answer/{token}` (token-authenticated, `phone.go:368`).

### Tests

- A `noauth` request carrying its own `X-Stream-App-Id` resolves to the proxy-set value, not the caller's.
- An `api_key` request with a valid key and a JWT signed by the wrong app secret is rejected.
- A revoked key stops working within the cache TTL.
- Every authentication failure returns the same 401 body — unknown key, bad signature, revoked key and stale token must not be distinguishable, or key ids can be enumerated.
- A scope failure returns 403, not 401.

### Files

- `acceleration/internal/auth/` — new: key lookup, JWT verification, the principal type
- `acceleration/internal/api/server.go` — `withCustomer` (`:196`) becomes mode-aware; `CustomerHeader` stays as the local-dev fallback
- `acceleration/internal/api/sessionws.go:31` — `CheckOrigin` returns `true` unconditionally; gate it on `ROUTER_CORS_ORIGINS`
- `acceleration/migrations/` — one new goose migration for `organizations`, `apps`, `api_keys`
- `acceleration/cmd/router/main.go` — read `ROUTER_AUTH_MODE`
- `acceleration/README.md` — the configuration table

---

## Workstream 2 — CI: the runtime image and the binary

**The release does not build in `Vision-Agents`.** That repo's CI keeps running tests and
nothing else; it is granted no access to Artifact Registry, to S3, or to any Stream cloud
account. The build runs in `chat`, which is already a trusted environment with the
credentials it needs, and checks out the acceleration source at a tag.

The flow is: tag in `Vision-Agents`, then dispatch the build in `chat`. Nothing about the
release path lives in the public repo, so there is no trust relationship to compromise
through a pull request, a workflow edit or a malicious dependency in that repo.

New workflow in `chat`, `workflow_dispatch` with a `version` input rather than a tag
trigger — a tag push in one repo cannot start a workflow in another, and a dispatch keeps
the trigger explicit and auditable. It checks out `GetStream/Vision-Agents` at
`accelerate-<version>` using `STREAM_CI_BOT_TOKEN`, which is also what lets `go mod
download` reach the private `getstream-go-webrtc` module through a `git config
url.insteadOf` rewrite.

**`binary` job** — matrix over `amd64` and `arm64` on chat's own `chat-<platform>-2404`
self-hosted runners, which already exist and are what `build-release-v2.yml` uses.
Install `pkg-config libopus-dev libopusfile-dev libsoxr-dev`, build with `CGO_ENABLED=1
go build -trimpath -ldflags="-s -w -X ...Version=$VERSION" -o router ./cmd/router`, emit
`router-linux-<arch>` and a `.sha256` sidecar, upload both to
`s3://stream-puppet/releases/r/GetStream/Vision-Agents/<version>/`. The sidecars are not
optional — shiply's preflight and the fetcher in Workstream 3 both read them.

**`image` job** — push the runtime shell to
`us-east1-docker.pkg.dev/${GCP_PROJECT_ID}/stream-services/accelerate`, multi-arch,
`provenance: false`, tagged with both the version and `sha-${GITHUB_SHA}`, with registry
build cache as `og` does.

Both jobs run on chat's **existing** release credentials — the Workload Identity binding
`build-release-v2.yml` already uses to push to Artifact Registry, and the
`AWS_*_CI_ARTIFACTS` secrets it already uses to `aws s3 cp` into `s3://stream-puppet/releases/`.
(`.github/actions/v2/s3-upload` is a different thing: it syncs CI artifacts to Hetzner.) This
workstream adds no new identity, no new binding and no new trust relationship anywhere.
That is the point: the only change to the credential graph is that one more workflow in an
already-trusted repo uses credentials that already exist.

Building the binary on the host rather than inside the image keeps the private-module token
out of the image build entirely — no `--mount=type=secret`, no layer to leak it in.

In `Vision-Agents`, the existing `go` job in `ci.yml` remains the test gate: it installs the
codecs and verifies the OpenAPI codegen is in step with the spec. It gains nothing new.

### Tests

- Dispatch a prerelease version and confirm four objects land in S3 and both platforms appear in `docker buildx imagetools inspect`.
- Confirm the checkout resolves the tag in the other repo and the private module fetches.
- Confirm `Vision-Agents` CI is unchanged: no new secret is referenced and no job touches a cloud account.

### Files

In `chat/`:

- `.github/workflows/build-accelerate.yml` — new; `workflow_dispatch` with a `version` input, cross-repo checkout, the two jobs above

In `Vision-Agents/`:

- `acceleration/Dockerfile` — two targets: `runtime` (the default, a shell with the codec libraries, the `nonroot` user, a digest-pinned base and `cmd/fetchbinary`, carrying no router) and `dev` (the same plus a router built in, which is what `compose.yaml` builds since there is no S3 locally)
- `compose.yaml` — `target: dev`, so local development is unchanged
- `acceleration/.dockerignore` — already excludes `deploy/`, `**/*_test.go`, testdata
- Nothing else. No new workflow, no new secret.

---

## Workstream 3 — the fetcher, in `chat`

Lives in `chat/projects/accelerate-fetch`, not here. The split is by ownership: Vision-Agents
produces the binaries, and everything about getting them onto our clusters — the fetcher, the
runtime image, the chart — belongs to the repository that owns the infrastructure. The service
should not have to know what a version registry is.

Its own small module rather than a command on `chat-manager`, which pulls in the whole
monolith: a ~130MB binary whose only job is to copy a file would be a strange initContainer.
`chat`'s `monolith/commands/manager/fetch_binary.go`, on the unmerged S3 branch, is the
reference for behaviour.

Resolve the version from
`s3://stream-services-version-registry/MultiRegion/us-east4/Accelerate/current` — the
**region-scoped** key form, which is what "enabled at the region, not the shard level"
means in `s3reg`'s key layout. Download the `.sha256` first, hash while streaming, write to
a temp file in the destination directory and `rename` into place, skip entirely when the
destination already matches, and record the resolved version to `<dest>.version`.
Credentials are a static AWS access key and secret, delivered as environment variables from
a Kubernetes Secret — **not** a GCP OIDC exchange for AWS STS. The chat branch federates
here; this deployment does not. The key belongs to an IAM user scoped to read exactly two
prefixes, the release prefix and the version registry, and nothing else.

Kubernetes owns the image; the version registry owns the binary. Do not put the version in
the pod spec. That single property is what stops a `rocky` deploy for an unrelated change —
a replica floor, a config value — from silently rolling the binary back to a stale release
while presenting as a healthy rollout.

Build it `CGO_ENABLED=0` so it stays small; it needs none of the codec libraries. The runtime
image carrying it is `chat/infra/docker/accelerate.Dockerfile`, which takes the cross-compiled
fetcher through a `bins` named build context exactly as `api.Dockerfile` takes the chat
binaries.

### Tests

- A checksum mismatch fails the fetch rather than installing a corrupt binary.
- A destination already matching the pointer is a no-op.
- A partially written download does not leave a runnable binary at the destination.

### Files

In `chat/`:

- `projects/accelerate-fetch/` — new module: the fetcher, its Makefile and its tests
- `infra/docker/accelerate.Dockerfile` — the runtime image carrying it and the codec libraries

In `Vision-Agents/`:

- `acceleration/Dockerfile` — stays a local-development image only, building the router and the gateway for `compose.yaml`. Production never uses it

---

## Workstream 4 — The Helm chart

`chat/infra/charts/accelerate/`, modelled on `chat/infra/charts/og/` — a service with its
own image, its own chart and a region-level toggle. Region-scoped install into namespace
`accelerate`, using `helm.ScopeValuesCluster`.

The values that matter:

- `replicas: 1`, `autoscaling.enabled: false`
- **No PodDisruptionBudget.** `chat/infra/charts/bus/values.yaml` states the rule plainly: a PDB on a single replica blocks node drains. `og`'s template self-suppresses below 2 replicas; borrow that guard or omit the template entirely
- securityContext from `chat/infra/charts/rocky-ui/values.yaml:160`, the strictest in that repo — `runAsNonRoot`, UID/GID 65532, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, `seccompProfile: RuntimeDefault`, and `readOnlyRootFilesystem: true` since the binary lands in an emptyDir and nothing else writes
- readiness `httpGet /health`, liveness `tcpSocket: 8080`, a startup probe with `failureThreshold: 12`
- Requests and limits both, memory limit equal to request, and `GOMEMLIMIT` set below the cgroup limit as `og` does so the GC gets aggressive before the OOM killer
- `maxUnavailable: 0`, `maxSurge: 25%`, `progressDeadlineSeconds: 1200`
- `terminationGracePeriodSeconds` raised — but note `cmd/router/main.go` hardcodes a 10s shutdown grace, so live WebRTC calls are cut at 10s no matter what the pod grace says. Raising it in the chart alone does not fix this; worth a follow-up
- ServiceMonitor carrying `release: kube-prometheus-stack`, or the operator will not adopt it

**Secrets, and no identity of our own.** Roughly thirty provider API keys, currently read
from a repo-root `.env`, plus the AWS access key the initContainer needs. They go into
Google Secret Manager through `rocky secrets set` — never `gcloud secrets` directly — and
are pulled in by External Secrets Operator against the `gcpsm` ClusterSecretStore with
`creationPolicy: Owner`. `STREAM_API_SECRET` is the highest-value one: it both signs inbound
webhooks and mints browser call tokens. On first install expect pods to crash-loop with
`secret not found` for up to two minutes while ESO syncs; that is documented behaviour, not
a chart bug.

The ServiceAccount is **plain** — no `iam.gke.io/gcp-service-account` annotation, no GSA, no
Terraform binding, no `aws.role_arn` and no projected token volume. Every credential the pod
holds is a static value that arrived through ESO. This diverges from `chat-api`, `og` and
`vep`, all of which annotate their service accounts; the divergence is deliberate and the
chart README should say so, or the next person to touch it will "fix" it.

One honest caveat: ESO itself authenticates to Secret Manager using the cluster's workload
identity, configured on the `bootstrap` chart's ClusterSecretStore. That is pre-existing
shared infrastructure used by every chart in the fleet, and this chart consumes it rather
than introducing it — but it does mean the path from pod to secret is not literally free of
workload identity end to end. Removing that too would mean abandoning ESO and creating
Kubernetes Secrets out of band, which trades a cluster-scoped binding for manual rotation
across thirty keys. Not recommended, but it is the remaining option if the standard has to
hold absolutely.

**NetworkPolicy.** Only one chart in the whole repo ships one, but it is not optional here:
the router runs in `noauth`, so ingress must be restricted to the proxy. Dataplane V2
enforces it.

**CronJob** for `POST /v1/stats/rollup`. Nothing in the process schedules it and the rollup
is idempotent.

### Tests

- `make -C infra test` — the `helm template` golden test, which CI's `prj/infra` job runs and which every chart in that repo is expected to bring
- Render with `replicas: 1` and assert no PodDisruptionBudget appears
- Assert the deployment omits `replicas` when the HPA owns it, so helm never fights the autoscaler
- Assert the rendered ServiceAccount carries **no** `iam.gke.io/gcp-service-account` annotation, and the pod spec has no projected token volume and no `AWS_ROLE_ARN` — a golden test is the only thing that will stop this being reintroduced by copy-paste from `og`

### Files

In `chat/`:

- `infra/charts/accelerate/` — `Chart.yaml`, `values.yaml`, `README.md` in the shape of `charts/rocky-ui/README.md`, and `templates/{_helpers.tpl,deployment.yaml,service.yaml,serviceaccount.yaml,externalsecret.yaml,servicemonitor.yaml,networkpolicy.yaml,cronjob.yaml}`
- `infra/shards/us-east4/accelerate.yaml` with `enabled: true` — its **absence** everywhere else is what disables it, per `Region.StackEnabled` in `infra/cli/internal/config/config.go`
- `infra/cli/cmd/accelerate.go` — a `rocky accelerate up` subcommand mirroring `infra/cli/cmd/og.go`
- `infra/cli/cmd/accelerate_chart_test.go` — the golden test
- No Terraform. There is no `google_service_account_iam_member` binding and no AWS role to create — that is the whole point of the change, and `rocky`'s `HasWorkloadIdentity` preflight must not be wired in for this chart or it will refuse to install

---

## Workstream 5 — The shiply descriptor

In `stream-infra`. Fetch and rebase `thierry/shiply-gke-deploy` on latest `main` first.

`shiply/internal/registry/services/accelerate.yaml`, validated against the
`service-schema.json` beside it:

```yaml
name: accelerate
binary_name: router
registry_name: Accelerate
scope: region
artifact:
  s3_path: s3://stream-puppet/releases/r/GetStream/Vision-Agents/{version}/{binary_name}-linux-{arch}
gke:
  kind: Deployment
  selector: app.kubernetes.io/name=accelerate
```

`scope: region` matches the chart toggle and produces the region-scoped registry key
`MultiRegion/us-east4/Accelerate/current` that Workstream 3 reads.

Nothing here touches puppet. The descriptor carries no `deploy_strategy: ssh`, no
`config_mode` and no `steps:` block — those are the EC2 path, and accelerate is GKE only, so
configuration comes from the Helm chart and External Secrets. The `stream-puppet` bucket
name is vestigial: it is still where both of chat's release workflows publish binaries on
master today, so it is the right bucket, but the name is the only puppet left in this.

Confirm before anything else that shiply will run this without a chat checkout —
`ValidateChatDir` and the hardcoded `S3ReleasePrefix` in `internal/aws/aws.go` are the two
things to check. `us-east4` already appears in `gcp_shards` in `registry.yaml`.

### Tests

- `go -C shiply test ./...`
- A dry run: `shiply deploy -s accelerate -S us-east4 -v v0.1.0 -n`

### Files

- `stream-infra/shiply/internal/registry/services/accelerate.yaml` — new
- `stream-infra/shiply/internal/aws/aws.go` — only if the hardcoded chat prefix turns out to block a non-chat service

---

## Workstream 6 — The proxy

A small Go service that terminates external traffic, authenticates, rate limits per
organization and app, and sets `X-Stream-Organization-Id` and `X-Stream-App-Id` on the way
through — stripping any inbound copy first.

This can land after the chart. The router in `noauth` behind a NetworkPolicy with no public
ingress is safe on its own; the proxy is what makes it reachable at all. Sequencing it
second gets something running in `us-east4` sooner.

WebSockets need explicit support: `/v1/agents/sessions/{id}/events`, `/v1/{modality}/stream`
and `/v1/dispatch` are long-lived with 30s pings and a 90s pong wait, so any idle timeout on
the proxy or the load balancer must exceed 90s or live calls drop.

### Tests

- A request arriving with its own `X-Stream-App-Id` has it replaced, not merged.
- A WebSocket held open past 90s idle survives.
- Rate limiting throttles one app without touching another in the same organization.

### Files

- `acceleration/cmd/gateway/main.go` — new
- `chat/infra/charts/accelerate/templates/` — the gateway deployment and the ingress

---

## Infrastructure prerequisites

Not code, and they gate the first deploy:

- **No new identity for `GetStream/Vision-Agents`.** It is granted nothing: no GCP binding, no Artifact Registry writer, no AWS role, no cloud credential in its repository secrets. The build runs in `chat` on credentials that already exist
- **An `accelerate` repository in Artifact Registry**, writable by chat's existing release credentials. Add it to `githubActionsArtifactRepos()` in `chat/infra/cli/cmd/gcp.go`
- **One AWS IAM user for the pod**, with a policy scoped to read `s3://stream-puppet/releases/r/GetStream/Vision-Agents/*` and the `Accelerate` key under `s3://stream-services-version-registry/MultiRegion/us-east4/`, and nothing else. Its access key goes into Secret Manager via `rocky secrets set`. Because it is a static credential it needs a named owner and a rotation schedule — record both at creation, since an unrotated key with no owner is the failure mode this choice trades for
- **Cloud SQL Postgres 18 and Memorystore for Valkey**, both in `chat/infra/shards/us-east4/accelerate.tf`, through the shared modules the shard leaves use. Sized for a canary: Postgres is ZONAL rather than REGIONAL, and Valkey runs with no replica — the file says what to raise first and why. Two GSM secrets (`shard-us-east4-accelerate-POSTGRES_PASSWORD` and `-POSTGRES_MIGRATOR`) must be seeded by `rocky secrets bootstrap` before Terraform runs, or the plan fails on a missing data source. No extensions are needed: the migrations are plain SQL, no pgvector, no custom types
- Bump `compose.yaml` from `postgres:16-alpine` to 18 for dev/prod parity
- A GCS or S3 bucket for `ROUTER_BLOB_URL`

---

## Verification

`sprint17.md` asks to tag, watch CI, iterate, then deploy step by step.

Tag the source, from `Vision-Agents`. This starts no build — that repo has no release
workflow and no credentials — it only names the commit:

```bash
git tag accelerate-v0.1.0 && git push origin accelerate-v0.1.0
```

Then dispatch the build from `chat`, which is where the credentials live:

```bash
gh workflow run build-accelerate.yml -R GetStream/chat -f version=v0.1.0
gh run watch -R GetStream/chat
```

Then confirm the artifacts exist, because shiply refuses the deploy if any are missing:

```bash
aws s3 ls s3://stream-puppet/releases/r/GetStream/Vision-Agents/v0.1.0/
docker buildx imagetools inspect us-east1-docker.pkg.dev/$GCP_PROJECT_ID/stream-services/accelerate:v0.1.0
```

Expect four objects — `router-linux-amd64`, `router-linux-arm64` and a `.sha256` for each —
and both platforms in the manifest.

Install the chart, from `chat/`. `rocky` renders a `helm diff` for approval before applying
anything:

```bash
make -C infra test
rocky accelerate up --region us-east4
```

Seed the version pointer by hand, once. Shiply refuses to deploy to a target whose pods have
no `fetch-binary` initContainer, which is exactly the state before the first install:

```bash
printf '%s' v0.1.0 | aws s3 cp - s3://stream-services-version-registry/MultiRegion/us-east4/Accelerate/current
```

Deploy:

```bash
shiply deploy -s accelerate -S us-east4 -v v0.1.0
```

Confirm, in order: the fetch worked, the service is healthy, a call completes.

```bash
kubectl -n accelerate logs <pod> -c fetch-binary | grep installed
kubectl -n accelerate port-forward svc/accelerate 8080:8080 && curl -s localhost:8080/health
```

`/health` should return 200 with `postgres` and `redis` both healthy in the `dependencies`
map — a 503 there names what is missing. Read the initContainer's logs rather than exec-ing
in; the runtime image has no shell. Finally place a call through the proxy and confirm the
agent answers with audio.

Rollback is `shiply deploy` of the previous version. `kubectl rollout undo` does **not**
work: it restores a pod spec, and the version is deliberately not in the spec.

---

## What changed during implementation

Recorded because the plan above says otherwise in places.

**The chart uses `strategy: Recreate`, not a rolling update.** The plan said
`maxUnavailable: 0`, `maxSurge: 25%`. That is wrong here: a surge starts the new pod before
the old one stops, and two routers overlapping is exactly the failure the single-replica
constraint exists to prevent — the second abandons the first's in-flight simulation runs.
The cost is a gap in service on every deploy, which is the honest trade until the router
holds its session state somewhere shared.

**No ServiceMonitor.** The plan asked for one. The router exposes no `/metrics` at all —
there is no prometheus or otel instrumentation in it — so it would scrape nothing and alert
on the failure. Adding `promhttp` to the router is the prerequisite.

**Shiply needed a schema change.** `deploy_strategy` was required and its only values were
`ssh` and `asg_replacement`; `ssh` pulls in required `build`, `discovery` and `steps` blocks
that a GKE-only service has none of. Added a third value, `gke`, plus a guard rejecting an
EC2 target for such a service — the mirror of the check the GKE path already makes. The
field is only ever compared against `asg_replacement` in code, so the new value is inert
everywhere else.

**The fetcher moved to `chat`, and the runtime image with it.** Vision-Agents produces the
`router` and `gateway` binaries and publishes them to S3; chat owns the fetcher, the image and
the chart. A consequence worth stating: the gateway is fetched from S3 too, by its own
initContainer resolving the same release pointer, so the two can never be a version apart.

**The gateway ships in the same chart.** It is pure Go and reaches no
private module, so it cross-compiles alongside `cmd/fetchbinary` in the runtime shell. It
is a separate Deployment rather than a sidecar: it is stateless and rolls normally, and
tying it to the router would inherit the one-pod limit for no reason.

## Open questions

- ~~**Does shiply run without a chat checkout?**~~ Answered: **no**. `resolveAllDeployParams` validates the chat directory before it branches on cloud, so a chat checkout has to be present even for a GKE-only deploy. Not a blocker — whoever runs shiply has one — but `-v` must be passed explicitly, since the interactive version picker lists chat's releases rather than Vision-Agents'.
- **What does the proxy authenticate against?** Whether it validates Stream app keys against chat's existing auth or holds its own store decides how much of Workstream 1's `api_key` mode the Stream deploy reuses.
- **`ROUTER_PUBLIC_URL` and telephony.** Vendor webhooks need a stable public HTTPS hostname. Internal-only means inbound telephony does not work in this deployment; confirm that is acceptable for this sprint, or it becomes a proxy requirement.
- **The dashboard is out of scope.** `NEXT_PUBLIC_ROUTER_URL` is inlined at build time, so its image is environment-specific and cannot be built once and promoted across environments. That has to be solved before the dashboard can be hosted.
