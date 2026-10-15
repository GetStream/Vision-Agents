---
paths:
  - "**/*.go"
---

# Documentation

## Inline Comments: Necessary and Sufficient

Comments must follow the math principle of **necessary and sufficient**: include only
what's required to understand the *why*, nothing more. Do not write walls of comments.

- **Necessary**: explain non-obvious intent, trade-offs, invariants, or constraints
  the code itself cannot convey.
- **Sufficient**: state it once, in the fewest words. No restating the code, no
  narrating history ("previously this did X"), no repeating the same point across
  paragraphs.

A comment longer than the code it explains is a red flag — collapse it to the
essential facts.

**Bad** (narrative, redundant, historical padding):
```go
// Retry the request up to three times so a transient network error doesn't
// fail the whole job. Previously this only retried once, which left the job
// failing under load whenever the upstream had a short blip.
//
// The upstream API doesn't accept idempotency keys, so a retry can repeat a
// side effect; capping retries at three keeps duplicates rare without
// pretending the call is safe to repeat.
```

**Good** (necessary + sufficient):
```go
// Retry up to three times so a transient network error doesn't fail the job.
// A retry can repeat a side effect (the upstream API has no idempotency keys).
```

Do NOT add comments that just narrate what the code does (`// increment counter`,
`// loop over users`). Delete them.

Do NOT reference planning docs, tickets, PR numbers, memory files, or skill
artefacts in code comments:

- No "Check #N", "plan step 7", "see `docs/plans/*`".
- No links to `~/.claude/projects/.../*.md` — those are private to one
  operator and don't exist for other readers.
- No "fix for PROJ-1234" or "see PR #4567" — that belongs in git log / PR
  body, not in source files.

These references drift the moment the plan / ticket / memory file changes
(or is removed). A comment must describe a constraint in the current code's
terms: what's the invariant, what's the trade-off, what's the surprising
fact a future reader needs. If the answer is "go read this external doc",
the comment is doing the wrong job.

Citing a public, stable source for a fact the code relies on is fine (an RFC, an engine's limits in
its official docs): the comment still states the fact, and the link is where it comes from.

## GoDoc Comments

New exported symbols need GoDoc comments as set out under "When Required" below.

### Format

Start with the symbol name, no dashes:

```go
// LoadConfig reads and validates the configuration file at path.
func LoadConfig(path string) (*Config, error) { ... }

// UserStore handles database operations for users.
type UserStore struct { ... }
```

**Bad:**
```go
// LoadConfig - reads and validates the configuration file
func LoadConfig(path string) (*Config, error) { ... }
```

### Multi-line for Complex Functions

```go
// Backoff returns the delay before the given retry attempt. The delay doubles
// with each attempt and is capped at max, so a long outage doesn't produce
// unbounded waits.
//
// The implementation adds full jitter, so clients that fail together don't
// retry in lockstep.
func Backoff(attempt int, base, max time.Duration) time.Duration { ... }
```

### When Required

- Exported functions called across package boundaries
- Exported types (structs, interfaces)
- Exported constants that affect behavior
- Any function with non-obvious logic

### When Optional

- Simple getters/setters
- Unexported (internal) functions
- Struct fields (struct-level doc is usually enough)
