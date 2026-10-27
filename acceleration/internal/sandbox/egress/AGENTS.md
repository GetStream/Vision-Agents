# internal/sandbox/egress

The router's egress proxy for code in a sandbox. The sandbox holds a placeholder in an env var; the proxy puts the connection's token in its place on the way out. The token never enters the sandbox. A host of the agent config's `allowed_domains` gets the sandbox's bytes as they are. `cmd/router` serves it on `code_sandbox.proxy_addr` (`ROUTER_CODE_SANDBOX_PROXY_ADDR`) when that is set.

When `code_sandbox.proxy_public_url` is `https://<public IPv4 address>:port`, that listener serves TLS, so the sandbox's proxy credential in `Proxy-Authorization` never crosses the network in the clear. Each handshake gets a certificate for that IP address (an IP SAN, no DNS name), signed by the router CA (`proxy_ca_cert`, `proxy_ca_key`), as each tunnel's is (`Proxy.ListenerTLS`). A plain-HTTP CONNECT on that port gets no answer, and a client that does not trust the CA completes no handshake. With `http://` or no public URL it serves plain HTTP.

## The TLS tunnel

Most of a sandbox's clients cannot use a TLS proxy URL (curl, urllib, wget, node; live probe, 2026-10-22). So with an `https` public URL the proxy broker gives the sandbox a forwarder, `tunnel.py`, an asyncio Python script with one thread (two threads on one SSL socket hang and crash: probe, 2026-10-22).

- It listens on `127.0.0.1:3128` and takes the clients' plain-HTTP proxy traffic. Each connection gets its own TLS connection to the proxy's address, verified with the router CA alone (`/tmp/va-egress-ca.pem`). It reads none of the bytes.
- `HTTPS_PROXY`, `HTTP_PROXY`, `https_proxy` and `http_proxy` are `http://sbx:<credential>@127.0.0.1:3128`, and `NODE_USE_ENV_PROXY=1`: node ignores the proxy env vars without it. `NO_PROXY` and the CA env vars are as with `http`.
- `Mount.Setup` writes the bundle, then starts the forwarder with `setsid nohup` and waits until it accepts. `Mount.EachRun` runs the same command before each run: a forwarder whose pid is alive and whose port accepts is left alone, since a restart cuts every download in flight. A dead one is started again.
- An image without `python3`, or another process on port 3128, fails the setup or the run, and no code runs.
- A connection that sends nothing, such as the check, opens no connection to the proxy.

## A request through it

1. The sandbox sends `CONNECT host:443` to the proxy, with `Proxy-Authorization: Basic` whose password is the sandbox's proxy credential. Every command and run of code gets the proxy in its env: `HTTPS_PROXY`, `HTTP_PROXY`, `https_proxy` and `http_proxy` are `http://sbx:<credential>@<proxy IPv4 address>:8091`, and `NO_PROXY`, `no_proxy` are `localhost,127.0.0.1,::1`. `GIT_HTTP_PROXY_AUTHMETHOD=basic` sets git's `http.proxyAuthMethod` (git-config(1)), so git sends the credential on its first CONNECT. Daytona's network allow list holds the proxy's address alone (`<address>/32`). So the sandbox reaches the proxy and, of Daytona's own network, only the netleash gateway (172.20.0.1:18080), by TCP or UDP, and has no DNS: the proxy looks up each host. Every sandbox with a network allow list can reach that gateway; with no `OutboundProxyUrl` it answers 403 (CIDR probe, a-gw and b-gw logs, 2026-10-22).
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
- The token's placeholder is in the body of every `ExecuteCommand` and `CodeRun` request to Daytona's toolbox, with the proxy credential (sdk-go v0.211.2 `process.go:159-160`, `234-235`): the sandbox's env travels with each command. A toolbox error that quotes its request would put them in the router's log and in `run_code`'s error. The token is never in them, only the placeholder and the proxy credential.
- Code that prints its env puts the proxy credential and the placeholders in the model's context. Neither is a token.
- A valid proxy credential works from wherever the proxy is reachable: the proxy cannot tell the sandbox from someone who copied the credential out of it. The accepted controls are the network and the credential's life, not code here: the port is reachable only from the sandbox provider's egress (AI-1582), and each sandbox gets its own random credential that expires with its session (`Store.Put`'s ttl) and ends on `Delete`.

## The broker

`Broker` (`broker.go`) is the `sandbox.CredentialBroker` for `code_sandbox.broker: proxy`.

- `Prepare` gives each grant a placeholder (`va_ph_<random>`) in its env var, stores one `Record` under a new proxy credential, and returns the proxy URL with that credential (`Mount.Proxy`, whose address is Daytona's network allow list), the CA certificate file, and the proxy and CA env vars. Those env vars are `Mount.RunEnv`, set on every command and run of code: Daytona's netleash replaces the ones a sandbox is created with, and not those (live probes, 2026-10-21). Only a connection that gives `sandbox.ErrNoCredential` leaves its grant out (or refuses a required one); one that cannot give a token just now keeps it, since the proxy asks on every request.
- The proxy's public URL (`code_sandbox.proxy_public_url`) is `http://` or `https://<public IPv4 address>:port` (`https` runs the TLS tunnel, see above), with a port from 1 to 65535 and no user, path, query or fragment. The router refuses to start with a host name, an IPv6 address, or an unspecified, loopback, private (RFC 1918), link-local (169.254.169.254 too) or broadcast address (`config.ProxyURL`): Daytona's network allow list takes IP ranges, the sandbox has no DNS, and it has no IPv6 route (live probe, 2026-10-22).
- Daytona gets no domain allow list and no `OutboundProxyUrl`. Either brings in its netleash (live probe, 2026-10-22). Under a domain allow list, netleash ends TLS itself and refuses the proxy's certificate for a granted host (staging E2E, 2026-10-22: `tls: bad certificate`).
- The CA file holds the setting's `CERTIFICATE` blocks alone. Node adds it to its roots (`NODE_EXTRA_CA_CERTS`); OpenSSL, Python requests and git replace theirs with a file, so `Mount.Setup` writes `/tmp/va-egress-ca-bundle.pem` from the image's own roots and the CA before any code, and fails, running no code, on an image with no roots file. Daytona's own image is Debian 13, whose roots are `/etc/ssl/certs/ca-certificates.crt` (150 roots, live probe 2026-10-21).
- A late grant (`sandbox.Grant.Late`) is one whose connection the user makes after the sandbox starts. `Prepare` gives it a placeholder without asking `Tokens`, and its record entry has no connection, so the proxy answers it 401 «gives no token», as for a disconnected one: a grant with no connection gets no token whatever `Tokens` would resolve (`Proxy.token`). Its host is reachable meanwhile, without a token. A late grant needs `BrokerOptions.Bind`, or `Prepare` fails.
- `BeforeRun` asks `Bind` for each late grant's connection and puts the connection and scheme it gives, alone, in the record entry, so the proxy sends the token from the next request on. It withdraws a grant whose connection needs a reconnect and stores the record again, which renews its TTL. With no grant left and no allowed domain, or after a `Close` the store refused, it deletes the record instead.
- `Close` deletes the record: the next request through the proxy is a 407.

## Metrics

`metrics.go` holds the proxy's Prometheus metrics, served on the router's `/metrics`.

- `router_sandbox_egress_requests_total{host, method, decision, status}` counts requests. A tunnel to a granted host is counted by the requests in it, not by its CONNECT. A tunnel to an allowed domain is counted by its CONNECT, since the proxy reads nothing in it. `status` is the class the sandbox got (`2xx`, `4xx`).
- `decision` is one of:
  - `token`: the placeholder was put in place of the token.
  - `granted_no_token`: a request to a granted host with no placeholder, sent as it is.
  - `blind`: a tunnel to an allowed domain.
  - `denied_host`: a host, a port or a tunnel's `Host` the sandbox has no grant for, or a grant that has ended.
  - `denied_auth`: no proxy credential, or an unknown or expired one (407).
  - `denied_method`: a method a granted host does not take.
  - `denied_request`: a request the proxy does not serve: not a CONNECT, a CONNECT with no port, or an upgrade.
  - `error`: the proxy could not serve it: the grants unreadable, a connection it could not take over, or a connection that gives no token.
- `router_sandbox_egress_upstream_seconds{host}` is the time a granted host took to answer a request, to its response header or its failure.
- `router_sandbox_egress_tunnels_open` is the tunnels open now, to granted hosts and to allowed domains.
- `host` is the entry of a grant's hosts that covers the request, as the connector's manifest wrote it (`*.example.com` is one value, whatever name under it is asked for), or `other` for any other host: an allowed domain, a refused host or a refused credential. `method` is one of RFC 9110's, else `other`. A token, placeholder, proxy credential, session or user is never a label value: the sandbox chooses its hosts and methods, and the label set must not grow with it.

## Tests

`proxy_test.go` (tag `integration`, needs `ROUTER_REDIS_ADDR`) runs the attack table A1-A14 of the sandbox broker waves, and reads the metrics after real requests, against a proxy whose hosts are `httptest` TLS servers and whose sandbox is a Go `http.Client` with the proxy URL. `broker_test.go` runs `brokertest` and the broker's own tests through the same proxy. `tunnel_test.go` runs the forwarder and its commands in a shell, under a directory that stands for the sandbox's root, against the proxy served over TLS (needs `python3`).
