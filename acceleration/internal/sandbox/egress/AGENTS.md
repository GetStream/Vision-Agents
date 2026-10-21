# internal/sandbox/egress

The router's egress proxy for code in a sandbox. The sandbox holds a placeholder in an env var; the proxy puts the connection's token in its place on the way out. The token never enters the sandbox. `cmd/router` serves it on `code_sandbox.proxy_addr` (`ROUTER_CODE_SANDBOX_PROXY_ADDR`) when that is set.

## A request through it

1. The sandbox sends `CONNECT host:443` with `Proxy-Authorization: Basic` whose password is its proxy credential (`http://sbx:<credential>@proxy:8091` in `HTTPS_PROXY`).
2. The proxy reads the sandbox's `Record` from Redis under the credential's SHA-256 (`Store`). None, or expired: 407.
3. The host must be a name one of the record's grants covers, and the port 443. An IP literal, any other host or any other port: 403, nothing is tunnelled.
4. The proxy answers the TLS handshake with a certificate for the host, signed by the router's CA (`code_sandbox.proxy_ca_cert`, `proxy_ca_key`). The sandbox trusts that CA's certificate.
5. For each request in the tunnel it reads the record again (a `Delete` ends open tunnels too), checks `Host` is the CONNECT host (else 421), and looks for a grant's placeholder in `Authorization`: `Bearer <ph>`, `token <ph>`, or Basic whose password is `<ph>`. Nowhere else: not the body, the query or another header.
6. With one, it asks `Tokens` for the token (the connections resolver, on every request: no copy is kept), puts it in the placeholder's place, and sends the request over TLS verified with the system's roots. The response's headers and body have the token, and Basic's encoded form of it, turned back into the placeholder.

## Rules

- The token is never logged, put in an error or returned to the sandbox. Logs name the session, binding, host and status only.
- A non-CONNECT request is refused (405): plain HTTP would carry a placeholder in the clear.
- Upgrades are refused (501): an upgraded connection's bytes would skip the response rewrite.

## Known limits

- A host that echoes the token in another encoding (escaped, compressed with an encoding the transport does not undo, split across headers) gives it to the sandbox. Grant only hosts that do not.
- A valid proxy credential works from wherever the proxy is reachable: the proxy cannot tell the sandbox from someone who copied the credential out of it. The accepted controls are the network and the credential's life, not code here: the port is reachable only from the sandbox provider's egress (AI-1582), and each sandbox gets its own random credential that expires with its session (`Store.Put`'s ttl) and ends on `Delete`.

## Tests

`proxy_test.go` (tag `integration`, needs `ROUTER_REDIS_ADDR`) runs the attack table A1-A14 of the sandbox broker waves against a proxy whose hosts are `httptest` TLS servers and whose sandbox is a Go `http.Client` with the proxy URL.
