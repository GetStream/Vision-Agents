# Do NOT

## Never Edit Generated Files

Files with a `Code generated ... DO NOT EDIT.` header are auto-generated.

Run the generator that produced the file instead of editing it.

## Never Commit Secrets

Do not commit:
- API keys or tokens
- Database connection strings with passwords

## Never Duplicate a Type That Exists Upstream

If a struct, slice, or constant already exists in a shared types package or another shared package, 
use it via narrow exports (helper functions, public getters) rather than defining a local copy. 
Doc-comments like "kept separate for stricter validation" or "decoupled from internal symbols" are usually
post-hoc rationale, not real constraints — stricter validation can run on
the canonical type just as well.

**Bad** — local copy with hand-maintained drift risk:

```go
// In cmd/tool/foo.go
type FooConfig struct {
    Threshold int    `json:"threshold"`
    Mode      string `json:"mode"`
}

var allowedFields = []string{"name", "type", "custom"} // mirrors internal/types/foo.go
```

**Good** — use the canonical type; if the upstream slice/map is unexported,
add a narrow exported helper next to it:

```go
// In cmd/tool/foo.go
var cfg types.FooConfig

// In internal/types/foo.go
func IsAllowedField(f string) bool { return slices.Contains(allowedFields, f) }
```

Acceptable narrow exceptions:

- The upstream type genuinely differs in shape (different JSON tags
  required, different field set). Document the difference explicitly.
- You need to evolve the type independently and breaking changes shouldn't
  ripple.

If neither applies, drop the duplicate.

## Never Add Defensive Code Without a Named Threat

Any defensive mechanism — transaction, `SELECT FOR UPDATE`, retry, fallback, post-write verification check, escape-hatch flag, skip-guard, "is the value still what we expected" reassertion — requires a `// Why: <concrete realistic scenario>` line naming the real threat it defends against.

If you can't name one, don't add it.

Examples that justify defensive code:
- Concurrent writers on the same resource (multiple server instances, multiple workers).
- Network failures where the caller needs idempotency.
- User input that can plausibly be malformed.

Examples that **don't** justify it:
- "Just in case."
- "For future-proofing."
- "To be safe."
