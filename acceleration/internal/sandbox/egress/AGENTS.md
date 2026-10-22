# internal/sandbox/egress

The router's egress proxy for code in a sandbox. The sandbox holds a placeholder in an env var; the proxy puts the connection's token in its place on the way out. The token never enters the sandbox. A host of the agent config's `allowed_domains` gets the sandbox's bytes as they are. `cmd/router` serves it on `code_sandbox.proxy_addr` (`ROUTER_CODE_SANDBOX_PROXY_ADDR`) when that is set.

## A request through it

1. The sandbox sends `CONNECT host:443` to the proxy, with `Proxy-Authorization: Basic` whose password is the sandbox's proxy credential. Every command and run of code gets the proxy in its env: `HTTPS_PROXY`, `HTTP_PROXY`, `https_proxy` and `http_proxy` are `http://sbx:<credential>@<proxy IPv4 address>:8091`, and `NO_PROXY`, `no_proxy` are `localhost,127.0.0.1,::1`. Daytona's network allow list holds the proxy's address alone (`<address>/32`). So the sandbox reaches the proxy and nothing else, by TCP or UDP, and has no DNS: the proxy looks up each host (live probe, 2026-10-22).
2. The proxy reads the sandbox's `Record` from Redis under the credential's SHA-256 (`Store`). None, or expired: 407.
3. The host must be a name one of the record's grants or allowed domains covers, and the port 443. An IP literal, any other host or any other port: 403, nothing is tunnelled. A host the allowed domains cover and no grant does gets a blind tunnel: the proxy dials it only when every address it resolves to is public (else 502), and copies bytes both ways. It reads none of them and puts in no token. A host both cover is a grant's, so steps 4 to 6 follow.
4. The proxy answers the TLS handshake with a certificate for the host, signed by the router's CA (`code_sandbox.proxy_ca_cert`, `proxy_ca_key`). The sandbox trusts that CA's certificate.
5. For each request in the tunnel it reads the record again (a `Delete` ends open tunnels too), checks `Host` is the CONNECT host (else 421), and looks for a grant's placeholder in `Authorization`: `Bearer <ph>`, `token <ph>`, or Basic whose password is `<ph>`. Nowhere else: not the body, the query or another header.
6. With one, it asks `Tokens` for the token (the connections resolver, on every request: no copy is kept), puts it in the placeholder's place, and sends the request over TLS verified with the system's roots. The response's headers and body have the token turned back into the placeholder: as sent, in a JSON string, URL-encoded, in base64 and in Basic's encoded form. `Range` and `If-Range` are dropped from the request, so no echo comes back in pieces. A response in a `Content-Encoding` the transport did not undo is refused (502), and its trailers and any 1xx response are not sent on.

## Rules

- The token is never logged, put in an error or returned to the sandbox. Logs name the session, binding, host and status only. An error from the host, or from a provider's refresh, may quote its answer, so only its class is logged; a dial or TLS error, which holds no byte of an answer, is logged as it is.
- A non-CONNECT request is refused (405): plain HTTP would carry a placeholder in the clear.
- Upgrades are refused (501): an upgraded connection's bytes would skip the response rewrite.

## Known limits

- A host that echoes the token in a form the rewrite does not know (another escape, split across headers, its case changed) gives it to the sandbox. Grant only hosts that do not.
- A host that keeps the token it was sent and returns it later, in the response to a request that carries no placeholder, gives it to the sandbox: only a response to a request with the token is rewritten.
- Sandbox code has root (passwordless `sudo`, live probe 2026-10-21). No control inside the sandbox protects anything: code can change its env, its CA files and its routes. The controls are the token, which stays in the router; Daytona's network allow list, which lets the sandbox reach the proxy's address alone (that code with root cannot bypass it by changing its routes, NET_ADMIN: unverified); and the grant check in step 3.
- A client that does not read the proxy env vars, such as a raw socket or a DNS lookup, reaches nothing. Plain HTTP through the proxy is refused (405).
- A `Delete` ends the tunnels to a grant's host. A blind tunnel to an allowed domain carries no token and lasts until either end closes it.
- A sandbox's proxy credential is in the env of every command and run of code, so sandbox code can read it and send it out to a granted host or an allowed domain. That is a proxy credential, never a token. It is the sandbox's own, it expires with the sandbox's record, and it reaches only the sandbox's grants and allowed domains. Daytona is not given it when it creates the sandbox.
- A valid proxy credential works from wherever the proxy is reachable: the proxy cannot tell the sandbox from someone who copied the credential out of it. The accepted controls are the network and the credential's life, not code here: the port is reachable only from the sandbox provider's egress (AI-1582), and each sandbox gets its own random credential that expires with its session (`Store.Put`'s ttl) and ends on `Delete`.

## The broker

`Broker` (`broker.go`) is the `sandbox.CredentialBroker` for `code_sandbox.broker: proxy`.

- `Prepare` gives each grant a placeholder (`va_ph_<random>`) in its env var, stores one `Record` under a new proxy credential, and returns the proxy URL with that credential (`Mount.Proxy`, whose address is Daytona's network allow list), the CA certificate file, and the proxy and CA env vars. Those env vars are `Mount.RunEnv`, set on every command and run of code: Daytona's netleash replaces the ones a sandbox is created with, and not those (live probes, 2026-10-21). Only a connection that gives `sandbox.ErrNoCredential` leaves its grant out (or refuses a required one); one that cannot give a token just now keeps it, since the proxy asks on every request.
- The proxy's public URL (`code_sandbox.proxy_public_url`) names an IPv4 address. The router refuses to start with a host name or an IPv6 address: Daytona's network allow list takes IP ranges, the sandbox has no DNS, and it has no IPv6 route (live probe, 2026-10-22).
- Daytona gets no domain allow list and no `OutboundProxyUrl`. Either brings in its netleash (live probe, 2026-10-22). Under a domain allow list, netleash ends TLS itself and refuses the proxy's certificate for a granted host (staging E2E, 2026-10-22: `tls: bad certificate`).
- The CA file holds the setting's `CERTIFICATE` blocks alone. Node adds it to its roots (`NODE_EXTRA_CA_CERTS`); OpenSSL, Python requests and git replace theirs with a file, so `Mount.Setup` writes `/tmp/va-egress-ca-bundle.pem` from the image's own roots and the CA before any code, and fails, running no code, on an image with no roots file. Daytona's own image is Debian 13, whose roots are `/etc/ssl/certs/ca-certificates.crt` (150 roots, live probe 2026-10-21).
- `BeforeRun` withdraws a grant whose connection needs a reconnect and stores the record again, which renews its TTL. With no grant left and no allowed domain, or after a `Close` the store refused, it deletes the record instead.
- `Close` deletes the record: the next request through the proxy is a 407.

## Tests

`proxy_test.go` (tag `integration`, needs `ROUTER_REDIS_ADDR`) runs the attack table A1-A14 of the sandbox broker waves against a proxy whose hosts are `httptest` TLS servers and whose sandbox is a Go `http.Client` with the proxy URL. `broker_test.go` runs `brokertest` and the broker's own tests through the same proxy.
