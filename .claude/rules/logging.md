---
paths:
  - "**/*.go"
---

# Logging

## Library

`log/slog` from the standard library.

## Error and Warning Logging

Use the context-aware methods, so the handler can tag the record with the trace ID:

```go
// Preferred — context-aware
logger.ErrorContext(ctx, "failed to process order", "error", err)
logger.WarnContext(ctx, "failed to refresh cache", "error", err)

// Without context (avoid when possible)
logger.Error("failed to process order", "error", err)
```

## Structured Fields

Put data in structured fields, not in the message string, and log through the logger rather than `fmt`/`log` (fields are indexed and queryable; see below):

```go
// Good — structured
logger.Info("order created",
    "order_id", orderID,
    "customer_id", customerID,
    "status", "pending",
)

// Good — single field
logger.Info("import started", "job_id", job.ID)

// Good — error field
logger.Warn("failed to create request", "error", err)

// Bad — unstructured
fmt.Printf("Error: %v\n", err)
log.Printf("Customer %s did something", customerID)
```

## Who Logs What

### HTTP Handlers
Handlers don't log an error they return. The server middleware logs it. A non-fatal side-effect failure that the handler swallows (e.g. failing to refresh a cache) is logged as a warning.

```go
// Good — return error, middleware logs it
if err != nil {
    return nil, err
}

// Bad — logging in handler
if err != nil {
    logger.Error("something failed", "error", err)
    return nil, err
}
```

### Data-Access Layer
The data-access layer logs warnings for **non-fatal** issues only:

```go
if err := s.cache.Invalidate(ctx, key); err != nil {
    if !errors.Is(err, cache.ErrNotFound) {
        logger.WarnContext(ctx, "failed to invalidate cache", "error", err)
    }
}
```

## Log Message Format

### Tracing Spans

Use `Type.Method` naming for tracing spans:

```go
// Data-access layer: "StructName.MethodName"
ctx, span := tracer.Start(ctx, "Store.GetOrder")
defer span.End()

// Handlers: "Handler.ActionName"
ctx, span := tracer.Start(r.Context(), "Handler.MyAction")
defer span.End()

// Standalone operations
subCtx, span := tracer.Start(ctx, "PrepareImportBatch")
defer span.End()
```

### Use Fields for Context, Not Tags in Messages

Use structured fields for filtering/querying — they're indexed in observability tools and support aggregation. Avoid embedding tags or metadata in the message string.

```go
// Good — structured fields, queryable in observability tools
logger := logger.With("component", "my-component")
logger.Error("batch processing failed",
    "batch_size", batchSize,
    "retries", retryCount,
)

// Bad — metadata buried in message string, not queryable as fields
logger.Error("ProcessBatch: batch processing failed, size=10, retries=3")
```

## Panic Recovery

Recover and log panics in goroutines:

```go
go func() {
    defer func() {
        if r := recover(); r != nil {
            logger.ErrorContext(ctx, "goroutine panicked", "panic", r, "stack", string(debug.Stack()))
        }
    }()
    // work
}()
```
