# internal/egress

The one way connector traffic leaves the router. Tenants supply the URLs connectors call, so every outbound request has to stay off private networks, the router itself and the cloud metadata server (server-side request forgery).

## API

- `ValidatePublicHTTPSURL(ctx, raw)` checks a URL before it is stored: `https` only, no userinfo, query or fragment, a port in 1–65535, no numeric host that is not an IP address, and every address the host resolves to public.
- `NewClient(timeout, wrap)` is the client for connector traffic. `wrap` is `Scheme.Wrap` and may be nil. It sits between the redirect policy and the address checks, so it can remove neither.
- `NewUnpooledClient(timeout, wrap)` is `NewClient` for the sandbox egress proxy (`sandbox/egress/proxy.go`): the same checks, and a request Go cannot send twice goes out on a new HTTP/1.1 connection (see the rules).
- `UserAgent` is the User-Agent the router sends when it fetches a URL someone else gave it (a photo, a voice note, a video clip). Hosts such as Wikimedia refuse Go's default one.
- `IsPublic(ip)` is the address check `NewClient` makes at dial. It is for traffic that is not HTTP, such as the SIP bridge's calls to a customer's trunk; the caller must check every address it connects to, not a name.

## Rules

- **Build clients only with `NewClient` or `NewUnpooledClient`.** Replacing the returned client's `Transport` or adding a proxy bypasses the address checks.
- **Check the IP that is dialed, not the name.** `dialPublic` resolves the host, refuses if any address is not public, and connects to the address it checked. That is what stops DNS rebinding; `TestTheClientDialsTheCheckedAddressNotTheName` and `TestANameThatRebindsToLoopbackAfterValidationIsRefused` lock it.
- **Redirects stay in the origin.** Same scheme, host and port (an empty `https` port is 443), at most `maxRedirects`, and never one that changes the method.
- **Only `NewUnpooledClient` keeps a request Go cannot send twice off kept-alive HTTP/1.1 connections.** An origin may close one as the next request arrives (its keep-alive limit, such as Apache's 5 s, ends first). Go sends a GET again on a new connection, but not a POST, nor a request whose body it cannot rewind, such as the one the sandbox proxy forwards. A client from `NewUnpooledClient` (the sandbox egress proxy's only; `checkedRoundTripper` finds `unpooled` set) sends those (`replayable` is false) through `unpooled`, which closes each HTTP/1.1 connection after one request and still shares HTTP/2 ones. `NewClient` stays pooled: every other caller would pay a handshake per POST for a race nobody measured. Never retry such a request instead: once its bytes left, the origin may have applied it. `TestAPostNeverRidesAConnectionTheOriginMayBeClosing`, `TestAPostTheOriginReadAndDroppedIsNotSentAgain` and `TestTheDefaultClientSendsPostsOnOneConnection` lock it.
- **Every refused prefix names its source.** Each entry in `nonPublicPrefixes` carries its IANA registry name and RFC in a trailing comment. Add entries only from the IANA special-purpose registries, with that comment.
- **New IANA entries come in through the snapshots.** Refresh `testdata/iana-ipv4-special-registry.csv` and `testdata/iana-ipv6-special-registry.csv` from https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry-1.csv and https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry-1.csv, with LF line endings. `TestEveryBlockIANAMarksNotGloballyReachableIsRefused` then names any block the list misses.
- **Tests never reach the public internet.** Public addresses are IP literals, and the resolver and dialer are fields of the unexported `policy`, so tests inject their own.

## Tests

From `acceleration/`: `go test -race ./internal/egress/...`. Testify suites, no mocks (`.claude/skills/go-testing/SKILL.md`).
