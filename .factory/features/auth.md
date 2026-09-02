# Authentication

Not asked for in any sprint yet, and not built. This is the research: what an API key and
secret should look like, how the secret should sign a request, and what creating and
removing a key has to get right.

## Where this starts

There is no authentication. [server.go](../../acceleration/internal/api/server.go) takes
`X-Customer-Id` at face value and says so in the comment beside it, and the sockets take
the same identifier as a `customer_id` query parameter because a browser WebSocket cannot
set a header. Every statistic, every voice, every agent config is keyed by that string, so
anyone who can reach the router can read and spend anyone else's.

One thing already signs: the Stream call webhook in
[callhooks.go](../../acceleration/internal/api/callhooks.go) verifies an HMAC over the body
before it will believe a delivery, because a hook without one is anyone who found the URL.
That is the same primitive this feature needs, applied to every request instead of one.

## The shape

Two values, with different jobs, which is the whole reason to prefer this over a single
opaque token:

- **The key id is public.** It travels in the clear, names which secret to verify with, and
  is what appears in logs, error reports, the dashboard and rate-limit counters. Naming a
  key in a log line is how an operator revokes the right one at three in the morning.
- **The secret never travels.** It is shown once at creation and then only ever used as an
  HMAC key. A request carries a signature, not the secret, so a proxy log, a browser
  history entry or a crash dump of the request line leaks nothing that can be replayed
  beyond the clock-skew window.

That second property is the one worth paying for. A bearer key in a header is a bearer
key: whoever reads it once holds it forever. A signature is worthless after five minutes.

### Format

```
key id   vak_live_7f3a9c2e5b1d8406
secret   vas_live_kQ8xR2vN...            43 chars, 256 bits of base64url randomness
```

- **Prefix both halves, and put the environment in the prefix.** `vak_`/`vas_` and
  `live`/`test` cost nothing and buy three things: a human can tell at a glance that a
  production secret has been pasted into a test config, a log scrubber can redact on the
  pattern, and GitHub's secret scanning partner program can be given a regex that finds
  leaked keys in public repositories. Partner patterns need a distinguishing keyword prefix
  precisely because a bare random string cannot be matched without false positives.
- **128 bits of entropy is the floor, 256 is the sensible default.** Generate with
  `crypto/rand`, not `math/rand`, and encode base64url so the value survives a URL, a shell
  and a YAML file unquoted.
- **Append a checksum.** Stripe and GitHub end their keys with a CRC32 of the body. It lets
  a client library reject a truncated paste locally, with a message that says the key is
  malformed, instead of turning it into a 401 that looks like a permissions problem.
- **Keep the last four characters in the clear.** The dashboard has to show a list of keys
  that a human can tell apart, and `…8406` plus a name and a creation date is enough.

## Storing the secret

Here is the tension nobody's blog post mentions. The received wisdom is right and does not
apply cleanly:

> Hash API keys with a salted SHA-256. High-entropy secrets get their strength from
> randomness, not from a slow hash, so bcrypt and Argon2 are a performance anti-pattern —
> a per-request cost measured in tens of milliseconds, on every call, forever. Reserve
> those for passwords.

But **a symmetric signature needs the secret back**. HMAC verification means recomputing
the signature, which means holding the key material, which means a hash at rest is not an
option. Choosing signing is choosing to keep recoverable secrets in the database. So:

- **Encrypt with AES-256-GCM under a key encryption key that lives outside the database.**
  Envelope encryption, KEK from KMS or from the environment, a `kek_version` column so the
  KEK can be rotated by re-wrapping rows without touching the secrets themselves. A
  database backup that leaks is then a leak of ciphertext.
- **Store a salted SHA-256 alongside it anyway.** It is what verifies a presented secret on
  the plain-bearer path (below), and it lets a lookup confirm a key without decrypting.
- **The alternative, if hash-at-rest matters more than client simplicity, is asymmetric.**
  Ed25519: the client keeps a private key, the server stores only the public half, and a
  database leak yields nothing signable at all. It is the strictly better answer on
  security grounds and the worse one on adoption grounds — every SDK and every curl example
  gets harder, and there is no equivalent of "paste this string into your config". Worth
  knowing it exists before committing to symmetric.

## Signing a request

Sign a canonical string, not the body. Signing only the body leaves the method, the path
and the query string free to be rewritten in flight, which means a captured `GET /v1/calls`
signature also authenticates `DELETE /v1/calls/123`.

```
VA1-HMAC-SHA256
<method>\n
<path>\n
<canonical query: sorted, percent-encoded>\n
<host>\n
<timestamp>\n
<nonce>\n
<hex sha256 of body>
```

```
Authorization: VA1-HMAC-SHA256 Credential=vak_live_7f3a9c2e5b1d8406, Signature=<hex>
X-VA-Timestamp: 1789012345
X-VA-Nonce: 01JQ8Z...
```

- **Version the scheme in its own name.** `VA1-` is four characters that make it possible to
  ship a second scheme later without a flag day. AWS does it with `AWS4-HMAC-SHA256`;
  Stripe does it with the `v1=` tag in the signature header, which also lets one delivery
  carry two signatures during a secret rotation.
- **Put the key id inside the signed string as well as in the header.** Otherwise the
  credential is unauthenticated and an attacker can swap it.
- **Include the host** if more than one environment can be reached with the same key, which
  is the point of the `live`/`test` split.
- **SHA-256, and nothing older.** SHA-1 is out. Hex or base64 is a taste question; hex
  greps better.
- **Compare with `hmac.Equal`.** A `==` on the signature is a timing oracle. Go's standard
  library already has the constant-time compare; the mistake is forgetting to reach for it.
- **Empty bodies hash to the known constant** `e3b0c442…b855`, so GET and POST take the same
  code path and there is no "if body is nil" branch to get wrong.
- **Sign nothing hop-by-hop.** Load balancers and proxies rewrite `Connection`,
  `Content-Length` and friends, and a signature over them fails in production and passes in
  every test.

### Replay

Timestamp and nonce do different halves of the job and neither is sufficient alone. A
timestamp bounds how long a captured request stays useful; a nonce stops it being used
twice inside that window.

- **A five-minute window.** Stripe's default tolerance, and the right trade-off. AWS allows
  fifteen, which is generous for a service whose clients are mostly servers with NTP.
  Reject timestamps that are too old; accept modest future skew rather than treating a
  client's fast clock as an attack.
- **Nonces in Redis, TTL equal to the window.** Redis is already a dependency, the storage
  is bounded by definition, and `SET NX` is the whole implementation. Without the timestamp
  the nonce set grows forever, which is why the two go together.
- **A Redis failure is a decision, not an accident.** Unlike the routing stats, where losing
  Redis means "no information", losing the nonce store means losing replay protection.
  Decide deliberately: fail closed on writes and open on reads, or fail open with an alarm.
  Write down which, and why, next to the code.

## Creating and removing a key

The lifecycle is where most of the operational value lives, and it is usually the part
built last and worst.

- **Show the secret exactly once.** The creation response is the only time it exists outside
  the client's hands. The dashboard must say so, loudly, before the dialog can be closed.
  No "reveal" button — a reveal button means the secret is retrievable, which means the
  encryption above bought much less than it looks like.
- **Several live keys per customer, always.** Rotation without downtime is: create the new
  key, deploy it, watch `last_used_at` on the old one go quiet, revoke. A one-key-per-account
  model forces every rotation to be an outage, so rotation stops happening.
- **Revocation is immediate and irreversible, and the row survives it.** Set `revoked_at`;
  do not `DELETE`. The audit trail of which key made which call has to outlive the key, and
  a deleted row turns a year of request logs into unattributable noise. Same shape as the
  soft delete on [voices](../../acceleration/migrations/20260826120100_voices.sql).
- **Record who did it.** Created by, revoked by, and when. Key creation is a privilege
  escalation primitive; an unaudited one is a backdoor.
- **`last_used_at` and `last_used_ip`, written lazily.** A synchronous update on every
  request is a write amplification disaster on a hot key. Throttle to once a minute per key,
  or push it through the same off-the-hot-path write the router already uses for request
  rows. Without it nobody can ever answer "is this key still in use?", and so nobody ever
  revokes anything.
- **Optional expiry.** `expires_at` with a warning before it lands. Mandatory expiry on a
  machine credential mostly produces 3am incidents; optional expiry with a nag produces
  hygiene.
- **Scopes, even if there are only two to start.** `read` and `write`, or per-modality. A
  key that can only read statistics is the one that goes in the CI dashboard, and adding
  scopes later means auditing every existing integration.
- **Rate limit per key, not per customer.** One runaway integration should throttle itself,
  not the account's production traffic.

### The table

```sql
CREATE TABLE api_keys (
    -- id is the public half: vak_live_..., what a request names and what a log line shows.
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    name TEXT NOT NULL,
    environment TEXT NOT NULL DEFAULT 'live',
    -- Signing needs the secret back, so it cannot be hashed. AES-256-GCM under a key
    -- encryption key held outside the database; kek_version lets that key be rotated by
    -- re-wrapping rows rather than reissuing secrets.
    secret_encrypted BYTEA NOT NULL,
    kek_version INT NOT NULL DEFAULT 1,
    -- secret_sha256 checks a presented secret without decrypting anything, for the bearer
    -- path and for cheap negative lookups.
    secret_sha256 BYTEA NOT NULL,
    -- last4 is all the dashboard ever shows of a secret again.
    last4 TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT '',
    expires_at TIMESTAMPTZ,
    -- Written lazily; a synchronous update here would double the writes of a busy key.
    last_used_at TIMESTAMPTZ,
    last_used_ip INET,
    -- Revoked keys are kept: a year of request rows referencing a deleted key is
    -- unattributable noise.
    revoked_at TIMESTAMPTZ,
    revoked_by TEXT NOT NULL DEFAULT ''
);

CREATE INDEX api_keys_customer_idx ON api_keys (customer_id, created_at DESC);
```

## Caching, and how fast a revocation takes effect

Decrypting and loading a key on every request is a database round trip in the hot path, so
it will be cached, and a cache is where revocation goes to die.

- **Cap the TTL at sixty seconds.** That is the honest promise: revocation takes effect
  within a minute.
- **Add a kill switch.** A Redis pub/sub invalidation that flushes a key from every
  process's cache, for the "we leaked one, get it out now" case where a minute is too long.

## What to say when it fails

- **One error for every authentication failure.** Unknown key id, wrong signature, revoked
  key, stale timestamp and reused nonce all return the same 401 with the same body. Telling
  a caller that the key id was valid but the signature was not is a free oracle for
  enumerating key ids.
- **401 for "I do not know who you are", 403 for "I know, and no".** A scope failure is a
  403 and can be specific, because the caller has already proved who they are.
- **Log the key id and never the secret or the signature.** Add both to whatever redaction
  the logger does, and test that.
- **Give the caller a request id.** Since the error is deliberately uninformative, support
  needs something to look the real reason up by.

## The two awkward callers

- **The sockets cannot sign.** A browser WebSocket sets no headers, and putting a
  credential in the query string spreads it through access logs, referrers and proxy
  traces — worse for a long-lived connection, where one leaked value hands over a live
  session. The pattern that works is a ticket: an authenticated HTTP call mints a
  single-use, ~30-second token bound to the customer and the intended socket, and the
  socket URL carries that instead. Smuggling the credential in
  `Sec-WebSocket-Protocol` is the other option, and it works, but it abuses the header and
  confuses every debugging tool that looks at it.
- **The dashboard is a browser and must never hold a secret.** It authenticates as a user —
  session or OIDC — and the server mints scoped short-lived credentials on its behalf. This
  is the model Stream itself uses: the secret stays server-side and signs a JWT per user,
  and the client only ever sees the API key and the token. Creating and revoking API keys
  is a dashboard action performed as a user, not as a key.

## The honest cost

Request signing is a real burden on every client: each SDK has to implement canonicalisation
identically, and "my signature does not match" is the worst class of integration bug because
the server cannot say why. TLS already gives integrity and confidentiality on the wire, so
signing earns its place through the things TLS does not do — the credential is never
transmitted, a captured request expires, and each request is individually attributable
end-to-end past any intermediary that terminates TLS.

The pragmatic answer is to support both and let the caller choose:

- `Authorization: Bearer vak_live_….vas_live_…` — the secret in the header, verified
  against `secret_sha256`. Trivial to adopt, fine over TLS, and what the first integration
  will actually use.
- `Authorization: VA1-HMAC-SHA256 …` — signed, required for the operations where replay or
  non-repudiation matters: creating keys, buying phone numbers, starting campaigns.

Starting signing-only means every early integration is a support ticket. Starting
bearer-only means signing never ships. Both, with the sensitive paths requiring the
stronger one, is the version that survives contact with users.

## Borrow, do not invent

- **AWS SigV4** is the fullest worked example — canonical request, string to sign, credential
  scope, a signing key derived through four chained HMACs so the key on the wire is scoped
  to a day and a service. The derivation is more than this needs; the canonicalisation
  rules are exactly what this needs.
- **Stripe's webhook signature** is the minimal version done well: `t=…,v1=…`, HMAC over
  `timestamp.body`, five-minute tolerance, multiple signatures per header during rotation,
  malformed entries skipped rather than fatal.
- **RFC 9421, HTTP Message Signatures**, is the standard answer, with `Signature-Input`
  naming the covered components including derived ones like `@method` and `@path`. It is
  worth reading and probably not worth adopting yet: adoption in 2026 is still emerging,
  Go library support is thin, and a custom scheme with a version prefix can migrate to it
  later.

## What to build first

1. `api_keys` as above, plus create, list and revoke on the dashboard and in the API.
2. Bearer verification middleware, replacing the `X-Customer-Id` header, with the customer
   id resolved from the key rather than asserted by the caller.
3. `last_used_at`, the audit columns, and per-key rate limiting.
4. The signed scheme, required on the paths that spend money.
5. The socket ticket, so `customer_id` leaves the query string.

The single largest security improvement is step 2 on its own. Everything after it is depth.

## Sources

- [API key best practices — Zuplo](https://zuplo.com/blog/api-key-best-practices)
- [Hashing and storage — apikeys.guide](https://apikeys.guide/docs/security/hashing-and-storage)
- [Password storage cheat sheet — OWASP](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [Why bcrypt will kill your API performance — Cyber Sierra](https://www.cybersierra.co/blog/bcrypt-performance-issues-api)
- [Signature Version 4 signing elements — AWS](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv4-signing-elements.html)
- [Receive Stripe events in your webhook endpoint — Stripe](https://docs.stripe.com/webhooks)
- [Protecting requests with HMAC, timestamps and nonces](https://thomasrones.com/technical/system-design/hmac-timestamp-nonce/)
- [HMAC secrets explained — GitGuardian](https://blog.gitguardian.com/hmac-secrets-explained-authentication/)
- [Secret scanning partner program — GitHub](https://docs.github.com/en/code-security/secret-scanning/secret-scanning-partner-program)
- [RFC 9421: HTTP Message Signatures](https://www.rfc-editor.org/rfc/rfc9421.html)
- [Essential guide to WebSocket authentication — Ably](https://ably.com/blog/websocket-authentication)
- [Tokens and authentication — Stream](https://getstream.io/chat/docs/javascript/tokens-and-authentication/)
