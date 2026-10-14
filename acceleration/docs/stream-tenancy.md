# Per-app Stream tenancy

What the router does in Stream for each customer, which app it does it in, and with which credential.

## The problem

The router held one Stream identity, `STREAM_API_KEY`/`STREAM_API_SECRET`, and made every Stream call with it, whoever called: conversation channels and replies, transcripts, call joins and SIP trunks, tokens and hook checks. On a shared router every customer's Chat and Video landed in the deployment's own Stream app, while the client SDKs open Chat and Video with the caller's own key. For any other app the agent joined a different call from the person, and replies landed where that app's clients never look.

## What it does

Every Stream call the router makes for a customer is made in that customer's own Stream app, resolved from the customer id alone. That includes the background work no request carries: pending writes, titles, campaigns, transfers and hooks. Work already written stays in the app it was written in, and is read back from there.

The router holds no customer's secret. The gateway in front (`stream-accelerate`) already holds every app's secret to check its callers, so it mints a 24-hour server token for the app it vouched for and forwards it with each request. The router keeps the newest one, sealed in Postgres, and acts in the app with it until it is about to expire. Whatever a server token cannot sign, an agent's or a guest's user token, the gateway's internal mint endpoint signs instead.

| Setting | Env | Default | Meaning |
|---|---|---|---|
| `stream.tenancy` | `ROUTER_STREAM_TENANCY` | `deployment` | `deployment`: everything in the env pair's app, as before. `app`: each customer in its own app, with the gateway's tokens |
| `stream.fallback` | `ROUTER_STREAM_FALLBACK` | `refuse` | App mode only. What a customer with no token gets: the deployment's app (`deployment`) or a refusal (`refuse`) |
| `stream.app_id` | `ROUTER_STREAM_APP_ID` | learned | The env pair's Stream app id. The router learns it from Stream at startup; setting it makes a mismatch refuse to start |
| `stream.mint_url` | `ROUTER_STREAM_MINT_URL` | empty | App mode only. The gateway's internal mint endpoint |
| `stream.mint_token` | `ROUTER_STREAM_MINT_TOKEN` | empty | App mode only. The bearer that endpoint expects |
| `auth.proxy_declares_kind` | `ROUTER_AUTH_PROXY_DECLARES_KIND` | false | Proxy mode: trust `Stream-Auth-Type` from the proxy, and read its absence as client-side (fail closed) |

App mode needs Postgres, a keyring (`ROUTER_AUTH_KEK`, or `ROUTER_AUTH_KEK_V<n>` with `ROUTER_AUTH_KEK_VERSION`), `auth.mode=proxy` with `auth.proxy_declares_kind`, and the mint endpoint. It refuses `STREAM_USER_TOKEN`. The deployment's own customer keeps the env pair's identity in both modes.

## How it works

- **`internal/streamapp`** is the one place an identity comes from.
  - A `Source` answers "which app, with which credential" for a customer, or for work pinned to an app. `Deployment` is the env pair. `Stored` is app mode: the deployment's own app for its own customer, then the customer's forwarded token, then the fallback.
  - `Forwarded` keeps tokens in `stream_app_tokens`, one per app, sealed with AES-GCM and bound to the app and its key. A token with under an hour left is refreshed at the mint endpoint.
  - `Clients` caches one Stream client per identity for 30 seconds, so a new token is used within that.
  - Nothing else in the router reads Stream credentials from the environment. A guard test keeps it that way.
- **What the gateway sends.** Only in proxy mode with `auth.proxy_declares_kind`, which is what lets the router believe it:
  - `X-Stream-Server-Token` and `X-Stream-Server-Token-Key` on every request it vouches for;
  - `X-Stream-Hook-App` on a hook whose signature it checked with the app's secret.
- **Pins.** Sessions, calls, phone numbers and call resources record the app they were made in (`stream_app_pk`): NULL is the deployment's app, `-1` an imported row this deployment cannot place, anything else a customer's own app. Reads and later writes follow the pin, never the caller's current app. A pin that can no longer be written parks, and is never delivered elsewhere.
- **Conversations.** Held in memory by customer and channel id, since a channel id is unique only within one app. One nobody holds finds its app from its session's pin.
- **Status.** `GET /v1/settings/app` reports the mode, where the app's work is written, its `stream_app_id`, and whether the `agent` channel and call types exist and are safe. The router creates neither.
- **Hooks.** A customer's hooks arrive at `/v1/chat/hooks/stream/{app}` and `/v1/phone/hooks/stream/{app}` through the gateway, which vouches for them. The deployment's own hooks are checked with its secret here. They are deduplicated, stale events are dropped, and bodies are bounded.
- **Floor.** An organisation can require its apps' own Stream app (`router stream-apps require`).
- **Operator CLI:** `router stream-apps require|legacy|fallbacks|backfill-pins`.

Apps that registered their keys before forwarded tokens were given a placeholder token row by migration `20261022120000`, so their background work asks the gateway for a token rather than waiting for their next request. Migration `20261023120000` then dropped the old `stream_apps` and `stream_app_keys` tables.

## Testing

From `acceleration/`:

```
go test -tags integration ./internal/streamapp ./internal/api ./internal/config ./cmd/router
go run ./cmd/openapi && git diff --exit-code -- api/openapi.yaml ../sdks/go/acceleration/generated.go
```

`StoredSuite` covers the source and forwarded tokens with a stand-in for the gateway's mint endpoint. `ForwardedTokensSuite` covers what the router believes of the gateway's headers. `TenancyGuardSuite` drives every path a customer takes in Stream and checks the deployment's key wrote nothing for a customer with an app of its own.

## Deploying

What app mode on a hosted router needs:
- **A keyring secret.** Postgres is already required.
- **The gateway** in front, forwarding the caller kind it verified (`jwt` for a token naming a user, `server` for one that does not) and each app's server token, with its internal mint endpoint reachable from the router only.
- **The `agent` channel and call types in each customer's app.**

```
ROUTER_STREAM_TENANCY=app
ROUTER_STREAM_FALLBACK=refuse
ROUTER_STREAM_APP_ID=<the deployment's app>
ROUTER_STREAM_MINT_URL=http://<gateway>-internal:8082
ROUTER_STREAM_MINT_TOKEN=<shared with the gateway's STREAM_ACCELERATE_INTERNAL_TOKEN>
```

**Rollback:**
- **Same binary:** `ROUTER_STREAM_TENANCY=deployment`, keep `ROUTER_STREAM_APP_ID`, restart. New work goes to the deployment's app, its old work keeps delivering, and customers' own apps' work parks instead of being misdelivered.
- **Migrations:** they are additive until the registration tables are dropped. Once a release has applied them, keep their filenames.

## Not covered yet

- Inbound Stream hooks and phone calls against real Stream.
- A region per app: `STREAM_BASE_URL` stays process-wide.
- Older Python SDKs.
