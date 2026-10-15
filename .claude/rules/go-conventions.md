---
paths:
  - "**/*.go"
---

# Go Conventions

## Naming

### Variables
```go
// Good - short names for limited scope
for i, v := range items { ... }
if err := doSomething(); err != nil { ... }

// Good - descriptive names for wider scope
var userCount int
var orderCache *sync.Map
```

### Receivers
```go
// Good - short, consistent (1-2 letters from type name)
func (c *Client) Send(req Request) error { ... }
func (s *Server) Start() error { ... }

// Bad - generic names
func (this *Client) Send(req Request) error { ... }
func (self *Server) Start() error { ... }
```

### Interfaces
```go
// Good - verb + "er" for single-method interfaces
type Reader interface { Read(p []byte) (n int, err error) }
type Sender interface { Send(req Request) error }

// Good - descriptive for multi-method interfaces
type OrderStore interface {
    Get(id string) (*Order, error)
    Save(o *Order) error
}
```

## Context

Always pass context as the first parameter:
```go
// Good
func GetUser(ctx context.Context, userID string) (*User, error)

// Bad
func GetUser(userID string, ctx context.Context) (*User, error)
```

Don't store context in structs that outlive the operation:
```go
// Bad - service outlives the context
type Service struct {
    ctx context.Context  // which request? already cancelled?
}

// OK - request-scoped struct with same lifetime as context
type RequestContext struct {
    Ctx  context.Context
    User *User
    ...
}
```

## Nil Handling

### Prefer nil slices
```go
// Good - nil is the zero value, no allocation
func GetOrders() []Order {
    if noOrders {
        return nil
    }
    return orders
}
```

Nil slices work identically to empty slices for `len()`, `cap()`, `range`, and `append()`.

**Exception**: Use empty slices for JSON/API responses where `[]` is preferred over `null`:
```go
// For JSON serialization
json.Marshal([]int(nil))  // "null"
json.Marshal([]int{})     // "[]"
```

### Check nil before dereferencing
```go
// Good
if user != nil && user.Profile != nil {
    name = user.Profile.Name
}
```

## Concurrency

### No goroutines in init()
```go
// Bad - runs at package load, untestable, no error handling
func init() {
    go backgroundTask()
}
```

### Prefer explicit Start() over launching in constructors
```go
// Good - user controls lifecycle
type Worker struct { ... }

func NewWorker(cfg Config) *Worker { ... }           // no goroutines
func (w *Worker) Start(ctx context.Context) error   // starts goroutines
func (w *Worker) Stop() error                       // cleanup

// Bad - user has no control over when goroutines start
func NewWorker(cfg Config) *Worker {
    w := &Worker{}
    go w.run() // surprising, hard to test
    return w
}
```

This follows `http.Server` pattern: create object, then call `ListenAndServe()`.

### Use channels for ownership transfer
```go
// Good - clear ownership
results := make(chan Result)
go func() {
    results <- computeResult()
    close(results)
}()
```

### Prefer sync primitives for simple state
```go
// Good for simple counters
var counter atomic.Int64
counter.Add(1)

// Good for protecting shared state
var mu sync.Mutex
mu.Lock()
defer mu.Unlock()
```

## Performance

### Use strconv over fmt for conversions
```go
// Good - faster
s := strconv.Itoa(42)
i, err := strconv.Atoi("42")

// Bad - slower, allocates more
s := fmt.Sprintf("%d", 42)
```

### Preallocate slices when size is known
```go
// Good
users := make([]User, 0, len(userIDs))
for _, id := range userIDs {
    users = append(users, getUser(id))
}

// Bad - multiple reallocations
var users []User
for _, id := range userIDs {
    users = append(users, getUser(id))
}
```

### Prefer pointer receivers for large structs
```go
// Good - avoids copying large struct
func (r *Report) UpdateSettings(s Settings) { ... }

// Fine for small, cheap-to-copy structs
func (p Point) Distance(other Point) float64 { ... }
```

## Struct Initialization

### Use field names in literals
```go
// Good - clear, resilient to field reordering
order := Order{
    ID:        "123",
    Status:    "pending",
    CreatedAt: time.Now(),
}

// Bad - breaks if fields are reordered
order := Order{"123", "pending", time.Now()}
```

### Zero values are valid defaults
```go
// Good - zero value is useful
var buf bytes.Buffer
buf.WriteString("hello")

// Good - explicit when zero value isn't enough
client := &http.Client{
    Timeout: 30 * time.Second,
}
```

## Maps

### Check existence when zero value is valid
```go
// Good
if val, ok := m[key]; ok {
    // key exists, val might be zero
}

// Bad - can't distinguish "not found" from "found zero"
if val := m[key]; val != 0 {
    // misses valid zero values
}
```

### Reading a nil map is safe; writing panics
```go
// Safe - reading from nil map returns zero value
var m map[string]int
v := m["key"] // v == 0, no panic

// Panic - writing to nil map
var m map[string]int
m["key"] = 1 // panic!
```

## References

- [Google Go Style Guide](https://google.github.io/styleguide/go/)
- [Uber Go Style Guide](https://github.com/uber-go/guide)
