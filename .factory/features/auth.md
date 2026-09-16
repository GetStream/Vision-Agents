# Authentication

[Sprint 17](../sprint17.md), "Auth", asked for two modes and built them. This document
describes the shape after that was revisited: four ways to decide who a caller is, an
access level on every endpoint, and three levels of end user that an app can narrow.

## Asked for

Four ways to answer "who is this", because four deployments have four different answers
already available to them:

- **`api_key`**, the default. The router verifies a key and a token signed with that key's
  secret. Several keys are live per app at once, and a token names the user it was minted
  for. This is the mode a customer running the router themselves uses.
- **`proxy`**. Something in front has already authenticated the caller and names them in
  headers: app, organization and user. The router verifies nothing and believes all three.
- **`noauth`**. Nothing is verified and nothing is asked for. Every caller is a backend.
- **custom**. The deployment embeds the module and supplies its own `Authenticator`.

On top of whichever mode is chosen: every endpoint is server-side only unless it says
otherwise, a server-side caller may act for a named user without being subject to that
user's permissions, and a client-side caller reaches its own sessions and no others. End
users come at three levels — anonymous, guest, and authenticated — all three of which may
open a session until an app says otherwise.

## The four modes

[internal/auth](../../acceleration/internal/auth) decides who a request is from, and
`ROUTER_AUTH_MODE` picks between the modes. The tables the first mode reads are in
[store/apps.go](../../acceleration/internal/store/apps.go), from
[the migration](../../acceleration/migrations/20260903120000_api_keys.sql).

`api_key` needs Postgres. The key travels in `X-Api-Key` and a token signed HS256 with that
key's secret in `Authorization: Bearer`; a socket carries both as query parameters, since a
browser WebSocket sets no headers. Proxy headers are ignored entirely here, because reading
them would be a way around the key. Every failure is the same 401 with the same body: a
caller that could tell an unknown key from a bad signature could use the difference to
enumerate key ids.

`proxy` believes what it is told. `X-Stream-App-Id` and `X-Stream-Organization-Id` name the
tenant, `X-Stream-User-Id` names the end user, and `Stream-Auth-Type` says which sort of
caller it is. Nothing is verified, so the proxy has to overwrite all four rather than
forward a caller's own. That proxy is `stream-accelerate`, which lives in the chat
repository because authenticating a Stream key means reading Stream's tables.

`noauth` asks for nothing. The tenant comes from `X-Customer-Id`, or from a `customer_id`
query parameter on a socket, and every caller is a backend acting for itself. It is for a
laptop, and the warning it logs on startup says so.

## Why `proxy` was split out of `noauth`

They were one mode, and calling it `noauth` was wrong in both directions. A deployment
behind Stream's proxy is not unauthenticated — somebody authenticated very carefully, just
not here — and a local deployment with no proxy was reading four headers it had no reason
to trust from anything that could reach it.

Keeping them together also meant the local case had to be the permissive one. Under one
mode, a caller on a laptop could set `Stream-Auth-Type: jwt` and become a client-side user
of any name, because the code could not tell a header the proxy had overwritten from a
header the caller had written. Split, `noauth` reads neither the user nor the auth type,
and there is nothing to spoof: everything is a backend because there is nothing that could
make it anything else.

The cost is a breaking change for one deployment. The split is shaped so that it is only
one: `noauth` keeps reading `X-Customer-Id`, which is what a local checkout sends, so a
laptop needs no change. What must move is the hosted router behind `stream-accelerate`,
which now sets `ROUTER_AUTH_MODE=proxy`.

## Why the default moved to `api_key`

The default was `noauth`, and a default that trusts whatever reaches it is only correct
when something else is guaranteeing that nothing does. That guarantee is a NetworkPolicy in
one deployment and nothing at all in the next, and the mode that needs it was the one you
got by saying nothing. `api_key` fails closed instead: without a store and a key encryption
key the router refuses to start, and it says which is missing rather than starting and
refusing every request for a reason only visible in a 401.

## Server-side only is the default

Every operation is server-side only unless the spec marks it `x-client-accessible`, and six
are: `createSession`, `listSessions`, `getSession`, `closeSession`, the session events
socket, and `search`. Those are the whole of holding a conversation and looking something
up, which is all an end user's device has any business doing. Everything else is refused
with a 403.

The check in front of the handlers reads the mark from the embedded spec rather than from a
list kept beside it, so what a generated SDK documents and what the server refuses cannot
drift apart. The default is that way round because the two mistakes do not cost the same:
an operation nobody thought about is refused to a browser, which arrives as a bug report,
where the other way round it is served to one, which arrives as a breach.

A caller says it is server-side twice. `Stream-Auth-Type: server` is the declaration and a
token carrying `server: true` with no `user_id` is the proof. Both are required because
each fails closed in a different direction: nothing signs the header, and a server token
pasted into a browser that sends the client header is treated as the browser it is. The
header deliberately has no query parameter counterpart, which is what stops a browser
opening a dispatch socket and answering somebody else's callers.

## A server-side caller may name a user

A backend is trusted completely: it holds the secret, so anything it could be refused it
could also mint a token for. Permissions are therefore not checked for it at all, and it
reaches every session its customer has.

It may still name a user, in `X-Stream-User-Id`. That header is not signed and does not
need to be, because the caller presenting it has already proved it is the customer's own
backend and could claim any user it liked by other means. What naming one buys is that the
session belongs to that user afterwards, so the user's own device can reach it: a backend
that opens a conversation on somebody's behalf and hands them the id is the ordinary way an
integration works, and without this the device would be refused its own session.

Naming a user does not cost the backend anything. It is still `KindServer`, still reaches
everything, and is still exempt from the daily limit — the limit counts end users, and a
backend given a user id is not one.

## Permissions: the session is the resource

For a client-side caller there is one rule, and it is about sessions: you reach your own
and nobody else's. `Owner.reaches` in
[session/manager.go](../../acceleration/internal/session/manager.go) is the whole of it.

Both halves of who a caller is have to match, not just the name. The kind is half of it
because an anonymous caller may go by any name it likes: without it, typing somebody else's
user id into a query parameter would be enough to read their conversation. So an anonymous
caller reaches only anonymous sessions bearing the same name, and a verified one reaches
only verified sessions.

Verified kinds reach a session their own backend opened for them. That is the one place the
rule is not strict equality: a session owned by `KindServer` and bearing a user id is
reachable by an authenticated or guest caller of that name. Anonymous is excluded, since an
anonymous name is a claim nobody checked and allowing it would let anyone read a
server-created session by guessing whose it was.

A session that exists but belongs to somebody else is reported as not existing at all,
because a 403 would confirm it was real. An anonymous caller that named nobody owns nothing
anybody else can be told apart from, so for those the session id is the whole of the
authority: it is random, and it is never listed.

## Three levels of end user

`auth.Kind` is what sort of caller a request is from, and it travels beside the user id
everywhere one person's things are kept from another's, because the id alone does not say
whether anybody checked it.

- **Anonymous** presented no token. Whatever name it goes by is unverified.
- **Guest** presented a verified token Stream issued for a temporary account. It is a real
  user, told apart from a permanent one only so a caller can be told which it was.
- **Authenticated** presented a verified token naming a user the customer knows.
- **Server** is not an end user at all; it is the customer's own backend.

All three end-user levels may open a session by default. An app can refuse anonymous
callers, guest callers, or both.

## Per-app features

The setting lives on the app, in a `settings` JSONB column added to the `apps` table. A
column of its own per toggle would mean a migration for each one, and these are the first
two of a list that is obviously going to grow; a document keyed by app is what the feature
actually is.

It is enforced where the caller is identified rather than in each handler. A level an app
refuses is not a caller that gets a narrower API, it is a caller that does not get in, and
doing it once at the door means a new endpoint cannot forget. That caller gets a 403 rather
than the 401 every other authentication failure gets, because it is the one failure reached
by somebody who has already proved who they are: nothing is given away by the distinction,
and the advice is the opposite one.

The default is permissive in both directions. A key that is not in the document allows,
which is what lets a setting be added without rewriting every row, and an app that resolves
to nothing allows, which covers `noauth` and `proxy` — neither reads these tables, so
neither has settings to read, and a `proxy` deployment's apps live in Stream's tables
anyway. The levels are therefore only enforced in `api_key` mode, which is the only mode
where the app is one of these rows.

The read costs nothing extra. It is joined into the same query that resolves the API key,
which was already happening on every authenticated request.

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

## Not done

- **Nothing creates a key but a database call.** `CreateOrganization`, `CreateApp`,
  `CreateAPIKey`, `ListAPIKeys` and `RevokeAPIKey` are all there and nothing outside a test
  calls them: no endpoints, no dashboard page, no bootstrap command. This is the gap that
  makes `api_key` mode awkward to actually adopt, and it is now the gap that makes the
  default awkward to adopt.
- **Nothing writes app settings either.** The column and the struct are there and are read
  on every request; turning a level off means an UPDATE by hand.
- **Permissions stop at the session.** Everything else a client-side caller may reach is
  decided by the endpoint's access level rather than by the row, which is enough only
  because the six client-accessible operations are all either session-scoped or read-only.
- **No scopes.** A key is all or nothing beyond the server-side split. Adding them later
  means auditing every integration that exists by then.
- **The credential is still in a socket's query string,** where it spreads through access
  logs and proxy traces. The fix is a single-use ticket minted by an authenticated HTTP call.
- **No per-key rate limiting, and no cache in front of the lookup.** Every authenticated
  request is a database read and an unseal. The proxy does the limiting in `proxy` mode;
  `api_key` has nothing.
