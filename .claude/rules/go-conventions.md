---
paths:
  - "**/*.go"
---

# Go Conventions

Follow the [Google Go Style Guide](https://google.github.io/styleguide/go/) and the
[Uber Go Style Guide](https://github.com/uber-go/guide). The points below are the ones that matter
most here.

## Context

- Pass `ctx context.Context` as the first parameter.
- Don't store a context in a struct that outlives the operation (a service, a client). A
  request-scoped struct with the same lifetime as the context may hold one.

## Lifecycle

- No goroutines in `init()`.
- A constructor starts no goroutines. Start them in an explicit `Start(ctx)` (or `Run(ctx)`) and stop
  them in `Stop()`/`Close()`, as `http.Server` does with `ListenAndServe()`.

## Nil Slices and JSON

Return a nil slice for "none", except in a JSON/API response, where `[]` is wanted over `null`:

```go
json.Marshal([]int(nil)) // "null"
json.Marshal([]int{})    // "[]"
```

## Maps

Use `v, ok := m[key]` when the zero value is a valid value. Writing to a nil map panics; reading
from one does not.
