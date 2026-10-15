---
paths:
  - "**/*.go"
---

# Timeouts and Deadlines

## Context Timeouts

All outbound calls (DB, HTTP, gRPC) must be bounded by a context timeout. Always `defer cancel()`.

```go
ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
defer cancel()

result, err := s.db.Query(ctx, query)
```

## HTTP Clients

Set transport-level timeouts — don't rely only on context:

```go
client := &http.Client{
    Transport: &http.Transport{
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
        TLSHandshakeTimeout:   10 * time.Second,
        ResponseHeaderTimeout: 10 * time.Second,
        ExpectContinueTimeout: 1 * time.Second,
        IdleConnTimeout:       90 * time.Second,
    },
}
```

Per-request timeout is controlled through context:

```go
reqCtx, cancel := context.WithTimeout(ctx, cfg.RequestTimeout)
defer cancel()
resp, err := client.Do(req.Clone(reqCtx))
```

## gRPC Clients

Use keepalive params for connection health. `grpc.DialContext` / `grpc.Dial` are deprecated in favour of
`grpc.NewClient`, which does no I/O at creation, so bound the first RPC with a context instead of the dial:

```go
conn, err := grpc.NewClient(host,
    grpc.WithTransportCredentials(creds),
    grpc.WithKeepaliveParams(keepalive.ClientParameters{
        Time:                10 * time.Second,
        Timeout:             1 * time.Second,
        PermitWithoutStream: true,
    }),
)

ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
defer cancel()
resp, err := client.Get(ctx, req)
```

## Server Middleware

When the HTTP server sets a request-level timeout in middleware, handlers don't need their own top-level timeout — but sub-operations (DB queries, external calls) should still have tighter timeouts.

## Never Unbounded

A request-path `ctx` carries the request deadline. The unbounded case is any context derived from
`context.Background()` or `context.TODO()` — background workers and goroutines detached from a request.
Every outbound call made from one of those needs its own `context.WithTimeout`, as in the example at the top.
