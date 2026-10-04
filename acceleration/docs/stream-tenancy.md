# Per-app Stream tenancy

Branch `nash/project-tenancy`. This is a summary for reviewers and for whoever tests and deploys it: what changed, how it was built, how to test it, and what a deployment needs.

## The problem

The router held one Stream identity, `STREAM_API_KEY`/`STREAM_API_SECRET`, and made every Stream call with it, whoever called. That covered:
- conversation channels and replies;
- transcripts;
- call joins and SIP trunks;
- tokens and hook checks.

So on a shared router every customer's Chat and Video landed in the deployment's own Stream app. The client SDKs, meanwhile, open Chat and Video with the caller's own key. For any other app:
- the agent joined a different call from the person;
- replies landed where that app's clients never look.

The agents quickstart's Agent UI step (`session.chat()`) cannot work on a shared router until this lands.

## What it does

Every Stream call the router makes for a customer is made in that customer's own Stream app, with that app's own key, resolved from the customer id alone. That includes the background work no request carries: the outbox, restart recovery, titles, campaigns, transfers and hooks. Work already written stays in the app it was written in, and is read back from there.

Two modes, chosen per deployment:

| Setting | Env | Default | Meaning |
|---|---|---|---|
| `stream.tenancy` | `ROUTER_STREAM_TENANCY` | `deployment` | `deployment`: everything in the env pair's app, as before. `app`: each customer in its registered app |
| `stream.fallback` | `ROUTER_STREAM_FALLBACK` | `refuse` | App mode only. What an unregistered customer gets: the deployment's app (`deployment`) or a refusal (`refuse`) |
| `stream.app_id` | `ROUTER_STREAM_APP_ID` | learned | The env pair's Stream app id. The router learns it from Stream at startup; setting it makes a mismatch refuse to start |
| `stream.deny_registration` | `ROUTER_STREAM_DENY_REGISTRATION` | empty | Customer ids that may not register |
| `stream.trust_api_key_header` | `ROUTER_STREAM_TRUST_API_KEY_HEADER` | false | Let a caller name which of its app's keys to act with |
| `auth.proxy_declares_kind` | `ROUTER_AUTH_PROXY_DECLARES_KIND` | false | Proxy mode: trust `Stream-Auth-Type` from the proxy, and read its absence as client-side (fail closed) |

App mode needs Postgres and a keyring: `ROUTER_AUTH_KEK`, or `ROUTER_AUTH_KEK_V<n>` with `ROUTER_AUTH_KEK_VERSION`. It refuses `STREAM_USER_TOKEN`.

The deployment's own app is never registered: its customer keeps the env pair's identity in both modes.

## How it works

- **`internal/streamapp`** is the one place an identity comes from.
  - A `Source` answers "which app, with which key" for a customer, or for work pinned to an app:
    - `Deployment` is the env pair.
    - `Stored` holds registered apps.
  - `Clients` caches one Stream client per identity, with a generation counter so a rotation or revocation drops it.
  - Nothing else in the router reads Stream credentials from the environment. A guard test keeps it that way.
- **Pins.** Sessions, calls, phone numbers and call resources record the app they were made in (`stream_app_pk`):
  - NULL is the deployment's app;
  - `-1` is an imported row whose app this deployment cannot place;
  - anything else is a registered app.
  - Reads and later writes follow the pin, never the caller's current app. A pin that can no longer be written parks, and is never delivered elsewhere.
- **Conversations.** Records for registered apps live under `CHAT_OUTBOX_DIR/apps/<hex customer>/`. The deployment's app keeps the old layout, so an older binary still finds its own records and never sees another app's.
- **Registration.**
  - `PUT /v1/settings/app/stream/credentials` takes every key the app holds.
  - The router asks Stream which app each key belongs to. It refuses:
    - a key of another customer's app;
    - an app that is suspended, or that does not check tokens;
    - the deployment's own app for anyone but its own customer.
  - Secrets are sealed with AES-GCM, bound to their row (customer, app, key), and never returned. Responses show the last four characters.
  - Disconnecting needs a key of the app as proof.
- **Status.**
  - `GET /v1/settings/app` reports the mode, where the app's work is written, the state and the keys.
  - It also reports whether the `agent` channel and call types exist and are safe. The router creates neither.
  - `POST /v1/settings/app/stream/check` re-checks against Stream now.
  - A background watch re-checks connected apps, and ends what a revoked or blocked app held.
- **Hooks.**
  - Stream hooks for a registered app arrive at `/v1/chat/hooks/stream/{app}` and `/v1/phone/hooks/stream/{app}`, verified with that app's secrets.
  - They are deduplicated, stale events are dropped, and bodies are bounded.
- **Floor.** An organisation can require its apps' own Stream app (`router stream-apps require`).
- **Operator CLI:** `router stream-apps register|list|check|rewrap|forget|require|legacy|fallbacks|backfill-pins`.
- **Go SDK:** `client.Settings().App`, `RegisterStream`, `DisconnectStream` and `CheckStream`.
- **The getstream CLI** (`agents sync`, and `play`/`test` with `--sync`) registers a project's key the first time it syncs to a router in app mode. Branch `nash/agents-stream-app` in that repo.

## With nothing set

A deployment that takes this build with no new setting stays in deployment mode. It keeps its Stream app, auth, token keys and outbox layout. New:
- one background `GET /app` at startup to learn the app id (logged; never blocks or fails startup);
- six additive migrations, run automatically.

Intended fixes that ride along:
- a call's transcript is read from the channel it was written to, within the call's window, and reading never creates a channel;
- users the router did not create are never overwritten (create-if-missing), so a chat-token `user_name` no longer renames an existing user;
- a campaign session joins the call it rang;
- hook bodies are capped at 1 MiB, or 4 MiB inflated;
- releasing a number deletes its trunk, and its routing rule for numbers attached by this build;
- validation errors no longer echo objects or secrets.

Also: `STREAM_HTTP_TIMEOUT` is no longer read, and Stream calls share one 30s client.

## How it was built

It was planned first: every Stream use site in `acceleration/`, which mode each runs in (request, background, hook), and what each needed. It was then built as small commits in four phases, each with its tests:

1. **Correctness fixes, no tenancy change:** transcript channel, user overwrites, campaign call ids, proxy caller kind, bounded hook bodies.
2. **The seam, behaviour unchanged:**
   - the `streamapp` resolver and client cache through every site;
   - pins on sessions, calls and numbers;
   - phone and conversations per app;
   - the env guard;
   - read-only `/v1/settings/app`.
3. **Stored credentials and app mode:**
   - one sealer gate on the shared keyring;
   - the key store, app mode and the fallback record;
   - the registration API, operator CLI and Go SDK.
4. **Hooks, floor and guard:**
   - per-app hooks;
   - `require_own_stream_app`;
   - an end-to-end guard that app mode never writes a registered customer into the deployment's app;
   - legacy counts;
   - the Go SDK authenticating whenever a customer id is set.

Then came a review pass (11 fixes), and a local end-to-end run against a real Stream test app. That run found three bugs, fixed in `4770d17a`, `8de9f858` and `f031a48e`:
- a 503 where an unregistered app should read "nowhere";
- an over-strict check of the `agent` channel type;
- misleading startup warnings in app mode.

Last came a simplification pass (9 commits). 64 commits in all.

## Testing

**Focused suites** from `acceleration/`:

```
go test ./internal/api ./internal/conversation ./internal/session ./internal/chatlog \
  ./internal/phone ./internal/store ./internal/streamapp ./internal/auth ./internal/config ./cmd/router
go run ./cmd/openapi && git diff --exit-code -- api/openapi.yaml ../sdks/go/acceleration/generated.go
```

`TestTurnRecordingSuite` fails at the base as well. `TestURLsSuite` fails only under the full parallel run.

**A local router in app mode** (scratch database, a Stream test app you can clean up):

```
ROUTER_ADDR=:18181 ROUTER_AUTH_MODE=noauth \
ROUTER_POSTGRES_DSN=postgres://postgres:postgres@localhost:55432/<scratch>?sslmode=disable \
ROUTER_REDIS_ADDR=localhost:56379 ROUTER_AUTH_KEK_V1=<any local string> \
ROUTER_STREAM_TENANCY=app ROUTER_STREAM_FALLBACK=refuse CHAT_OUTBOX_DIR=<scratch dir> \
BASETEN_API_KEY=<any value; session start needs an LLM route> ./router
```

Then, with `X-Customer-Id: <app id>`:
1. `GET /v1/settings/app` reads `nowhere`.
2. A text session is refused.
3. `PUT …/stream/credentials` connects the app.
4. A session's channel is created in that app.
5. The chat token carries the app's key.
6. Guests are refused until the app allows them.
7. Disconnecting with proof works, and new sessions are refused again.

Delete the channels and users the run made.

**Before this reaches `accelerate`**, in order:
1. Rebase onto `accelerate` and renumber the six migrations after its newest. Today they sit between migrations already there, and goose refuses either order.
2. CI on the pushed branch: `ci.yml` runs on every branch.
3. Make the end-to-end run above into committed, env-gated tests over three apps:
   - A, the deployment's app;
   - B, registered;
   - C, unregistered.
   - They also cover restart recovery into B, key rotation, revocation, and same-binary rollback.
4. Run the real clients against a local router: the getstream CLI quickstart, the dashboard's Join call and transcript, and the docs panel.
5. A release candidate on the hosted environment, built from the branch with the launch tooling:
   - back up the database first;
   - deploy once with nothing set, then switch to app mode;
   - run the suite from step 3;
   - drill the rollback.
6. Merge the same commits, release, and deploy the release.

## Deploying

What app mode on a hosted router needs:
- **A keyring secret.** Postgres is already required.
- **The gateway forwards the caller kind it verified**, and `ROUTER_AUTH_PROXY_DECLARES_KIND=true` is set:
  - `jwt` for a token naming a user;
  - `server` for one that does not;
  - never `server` for an app whose tokens are not checked.
  - Without it, the router refuses every registration behind a proxy, and browsers keep reaching server-only operations.
- **Anything that calls the router directly,** without the gateway, names its app with `X-Stream-App-Id`. With the setting above, proxy mode no longer reads `X-Customer-Id`.
- **A durable `CHAT_OUTBOX_DIR`** (recommended). On an emptyDir every restart loses each conversation's command ledger.
- **The `agent` channel and call types in each registered app.**

Settings:
```
ROUTER_STREAM_TENANCY=app
ROUTER_STREAM_FALLBACK=refuse
ROUTER_STREAM_APP_ID=<the deployment's app>
```

The deployment's own app needs no registration. Every other app registers through `agents sync`, the dashboard, or `router stream-apps register`.

**Rollback:**
- **Same binary:** `ROUTER_STREAM_TENANCY=deployment`, keep `ROUTER_STREAM_APP_ID`, restart. New work goes to the deployment's app, its old work keeps delivering, and registered apps' work parks instead of being misdelivered.
- **An older binary:** it ignores pins and never reads `apps/`.
- **Migrations:** they are additive. Once a release has applied them, keep their filenames.

## Not covered yet

- Inbound Stream hooks and phone calls against real Stream.
- More than one replica: the caches and dispatch record are per process.
- A region per app: `STREAM_BASE_URL` stays process-wide.
- Older Python SDKs.
