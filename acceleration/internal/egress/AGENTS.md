# internal/egress

The one way connector traffic leaves the router. Tenants supply the URLs connectors call, so every outbound request has to stay off private networks, the router itself and the cloud metadata server (server-side request forgery).

## API

- `ValidatePublicHTTPSURL(ctx, raw)` checks a URL before it is stored: `https` only, no userinfo, query or fragment, a port in 1–65535, no numeric host that is not an IP address, and every address the host resolves to public.
- `NewClient(timeout, wrap)` is the only client for connector traffic. `wrap` is `Scheme.Wrap` and may be nil. It sits between the redirect policy and the address checks, so it can remove neither.
- `NewRoute(proxy)` is `NewClient` and `DialPublic` through an HTTP CONNECT proxy, or direct when `proxy` is nil. The router builds connector clients with it from `connectors.egress_proxy`. Every address is checked before the proxy is dialed, and the CONNECT names the target's host and port.
- `UserAgent` is the User-Agent the router sends when it fetches a URL someone else gave it (a photo, a voice note, a video clip). Hosts such as Wikimedia refuse Go's default one.
- `IsPublic(ip)` is the address check `NewClient` makes at dial. It is for traffic that is not HTTP, such as the SIP bridge's calls to a customer's trunk; the caller must check every address it connects to, not a name.

## Rules

- **Build clients only with `NewClient` or a `Route`.** Replacing the returned client's `Transport` or setting its `Proxy` bypasses the address checks. A proxy comes only from the router's own setting through `NewRoute`, never from `HTTPS_PROXY`.
- **Through a proxy, check first, then tunnel.** `dialPublic` refuses a non-public target before it dials the proxy, so a private address never reaches it (`TestThroughAProxyANonPublicTargetIsRefusedBeforeAnyConnect`). The CONNECT names the host, not the checked IP, so the proxy can apply an allow-list by name; it resolves the name again, so it must refuse non-public addresses itself. TLS stays end to end and is verified against the name.
- **Check the IP that is dialed, not the name.** `dialPublic` resolves the host, refuses if any address is not public, and connects to the address it checked. That is what stops DNS rebinding; `TestTheClientDialsTheCheckedAddressNotTheName` and `TestANameThatRebindsToLoopbackAfterValidationIsRefused` lock it.
- **Redirects stay in the origin.** Same scheme, host and port (an empty `https` port is 443), at most `maxRedirects`, and never one that changes the method.
- **Every refused prefix names its source.** Each entry in `nonPublicPrefixes` carries its IANA registry name and RFC in a trailing comment. Add entries only from the IANA special-purpose registries, with that comment.
- **New IANA entries come in through the snapshots.** Refresh `testdata/iana-ipv4-special-registry.csv` and `testdata/iana-ipv6-special-registry.csv` from https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry-1.csv and https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry-1.csv, with LF line endings. `TestEveryBlockIANAMarksNotGloballyReachableIsRefused` then names any block the list misses.
- **Tests never reach the public internet.** Public addresses are IP literals, and the resolver and dialer are fields of the unexported `policy`, so tests inject their own.

## Tests

From `acceleration/`: `go test -race ./internal/egress/...`. Testify suites, no mocks (`.claude/skills/go-testing/SKILL.md`).
