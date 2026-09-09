# Authentication

[Sprint 17](../sprint17.md), "Auth". Built, minus a way to create a key that is not a
database call.

## Asked for

Two modes. `noauth`, for a deployment where a proxy in front already limits who can reach
the router, and `api_key`, where the router verifies a key and secret itself. Keys belong to
apps, apps belong to organizations, and every caller on the API is checked.

## What exists

[internal/auth](../../acceleration/internal/auth) decides who a request is from and
[store/apps.go](../../acceleration/internal/store/apps.go) holds the three tables from
[the migration](../../acceleration/migrations/20260903120000_api_keys.sql).
`ROUTER_AUTH_MODE` picks between them, defaulting to `noauth`.

`noauth` believes what it is told: `X-Stream-App-Id` and `X-Stream-Organization-Id`, or
`X-Customer-Id` alone, and the same as a `customer_id` query parameter on a socket. Nothing
is verified, so the proxy has to overwrite both headers rather than forward a caller's own.
That proxy is `stream-accelerate`, which lives in the chat repository because authenticating
a Stream key means reading Stream's tables.

`api_key` needs Postgres. The key travels in `X-Api-Key` and a token signed HS256 with that
key's secret in `Authorization: Bearer`; a socket carries both as query parameters, since a
browser WebSocket sets no headers. The proxy headers are ignored entirely here, because
reading them would be a way around the key. Every failure is the same 401 with the same
body: a caller that could tell an unknown key from a bad signature could use the difference
to enumerate key ids.

## Why a token and not a signed request

The research this document used to be laid out a full AWS-SigV4-shaped scheme —
`VA1-HMAC-SHA256` over a canonical string of method, path, query, host, timestamp, nonce and
body hash, with the nonces in Redis. That is not what shipped, and the reason is adoption.
Stream's own server SDKs already mint an HS256 token from an app secret, so a customer's
existing code fits with no canonicalisation to get identically right in five languages, and
"my signature does not match" is the worst class of integration bug because the server
cannot say why. What signing buys over a bearer secret is that the secret never travels and
a captured request expires; a token with a required `exp` buys both of those too.

## The secret cannot be hashed

The usual advice is to hash an API secret and never hold it back. It does not apply, and
this is the one place the shape of the feature is forced: verifying a token means
recomputing its signature, which means holding the key material. So secrets are sealed with
AES-256-GCM under `ROUTER_AUTH_KEK`, a key outside the database, and `kek_version` is there
so it can be rotated by re-wrapping rows rather than reissuing every secret. A leaked backup
yields ciphertext.

The rest of the credential's shape is borrowed rather than invented. `vak_live_…` and
`vas_live_…` put the environment in the prefix so a production secret pasted into a test
config is visible as one, and the key id ends in a CRC32 so a truncated paste is rejected
before the database is asked. `last4` is all the dashboard would ever show of a secret
again. A revoked key keeps its row, because a year of request rows pointing at a deleted key
is unattributable noise, and `last_used_at` is written at most once a minute per key, since
a synchronous update would double the writes of a busy one.

## Server-side only

Thirty-three paths configure the agent rather than talk to it, and are marked
`x-server-side-only` in [the spec](../../acceleration/api/openapi.yaml), which is where both
the generated SDKs and the check in front of the handlers read it from. A caller says it is
one twice: `Stream-Auth-Type: server` is the declaration and a token carrying `server: true`
with no `user_id` is the proof. Both are required because each fails closed in a different
direction — nothing signs the header, and a server token pasted into a browser that sends
the client header is treated as the browser it is. The header deliberately has no query
parameter counterpart, which is what stops a browser opening a dispatch socket and answering
somebody else's callers.

## Not done

- **Nothing creates a key but a database call.** `CreateOrganization`, `CreateApp`,
  `CreateAPIKey`, `ListAPIKeys` and `RevokeAPIKey` are all there and nothing outside a test
  calls them: no endpoints, no dashboard page, no bootstrap command. This is the gap that
  makes `api_key` mode awkward to actually adopt.
- **No scopes.** The table has no column for them, so a key is all or nothing beyond the
  server-side split. Adding them later means auditing every integration that exists by then.
- **The credential is still in a socket's query string,** where it spreads through access
  logs and proxy traces. The fix is a single-use ticket minted by an authenticated HTTP call.
- **No per-key rate limiting, and no cache in front of the lookup.** Every authenticated
  request is a database read and an unseal. The proxy does the limiting in `noauth`;
  `api_key` has nothing.
