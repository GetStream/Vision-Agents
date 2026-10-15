---
paths:
  - "**/*.go"
---

# Error Handling

## Error Wrapping

Wrap with `fmt.Errorf` and the `%w` verb, so callers can still match the cause with `errors.Is` / `errors.As`.
Start the message with the package name and say what was being done, without "failed to", as the
router does (`oauth2cc: token: %w`):

```go
// Wrap with additional context at boundaries
fmt.Errorf("store: get user %s: %w", userID, err)

// Create new error
errors.New("store: no such user")
fmt.Errorf("store: invalid input %q", input)
```

### When to wrap vs return as is

- **`return err`** — when the original error is self-explanatory (e.g., it already says "connection refused")
- **`fmt.Errorf("...: %w", err)`** — at layer boundaries where caller context adds debugging value (e.g., which user, which order, which operation)

```go
// Good — adds useful context at a boundary
func (s *Store) GetOrderItems(ctx context.Context, orderID string) ([]Item, error) {
    items, err := s.db.Query(ctx, query, orderID)
    if err != nil {
        return nil, fmt.Errorf("store: get items for order %s: %w", orderID, err)
    }
    return items, nil
}

// Good — inner error is already descriptive enough
if err := validateInput(req); err != nil {
    return nil, err
}
```

## Handle Errors Once

Never log and return the same error — it creates duplicate noise in observability:

```go
// Bad — error appears twice in logs
if err != nil {
    logger.Error("failed to get user", "error", err)
    return nil, err
}

// Good — return it, let the caller (or middleware) log
if err != nil {
    return nil, err
}
```
