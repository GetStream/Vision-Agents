# internal/egress

The one way connector traffic leaves the router. Tenants supply the URLs connectors call, so every outbound request has to stay off private networks, the router itself and the cloud metadata server (server-side request forgery).

## API

- `ValidatePublicHTTPSURL(ctx, raw)` checks a URL before it is stored: `https` only, no userinfo, query or fragment, a port in 1–65535, no numeric host that is not an IP address, and every address the host resolves to public.
- `NewClient(timeout, wrap)` is the only client for connector traffic. `wrap` is `Scheme.Wrap` and may be nil. It sits between the redirect policy and the address checks, so it can remove neither.

## Rules

- **Build clients only with `NewClient`.** Replacing the returned client's `Transport` or adding a proxy bypasses the address checks.
- **Check the IP that is dialed, not the name.** `dialPublic` resolves the host, refuses if any address is not public, and connects to the address it checked. That is what stops DNS rebinding; `TestTheClientDialsTheCheckedAddressNotTheName` and `TestANameThatRebindsToLoopbackAfterValidationIsRefused` lock it.
- **Redirects stay in the origin.** Same scheme, host and port (an empty `https` port is 443), at most `maxRedirects`, and never one that changes the method.
- **Every refused prefix names its source.** Each entry in `nonPublicPrefixes` carries its IANA registry name and RFC in a trailing comment. Add entries only from the IANA special-purpose registries, with that comment.
- **New IANA entries come in through the snapshots.** Refresh `testdata/iana-ipv4-special-registry.csv` and `testdata/iana-ipv6-special-registry.csv` from https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry-1.csv and https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry-1.csv, with LF line endings. `TestEveryBlockIANAMarksNotGloballyReachableIsRefused` then names any block the list misses.
- **Tests never reach the public internet.** Public addresses are IP literals, and the resolver and dialer are fields of the unexported `policy`, so tests inject their own.

## Tests

From `acceleration/`: `go test -race ./internal/egress/...`. Testify suites, no mocks (`.claude/skills/go-testing/SKILL.md`).
