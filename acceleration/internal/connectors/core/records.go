package core

import (
	"context"
	"errors"
	"net"
	"sync"
	"time"
)

// Correlation names what caused a piece of connector work, for the audit row it leaves: the
// API request it came in on, and the session whose tool call it served. Either is empty when
// nothing of that kind caused it, and SessionID is empty for an incognito session, whose
// calls are tied to no conversation.
type Correlation struct {
	RequestID string
	SessionID string
}

type correlationKey struct{}

// WithCorrelation is ctx carrying c, which replaces what ctx carried.
func WithCorrelation(ctx context.Context, c Correlation) context.Context {
	return context.WithValue(ctx, correlationKey{}, c)
}

// CorrelationOf is what WithCorrelation put on ctx, zero when nothing did.
func CorrelationOf(ctx context.Context) Correlation {
	c, _ := ctx.Value(correlationKey{}).(Correlation)
	return c
}

// Exchange is what a connection's client saw of the requests one tool call sent, read by the
// session's dispatcher to say how the call failed (store.ConnectorInvocation.ErrorType): a
// tool source's error does not say whether a request left, and an MCP client reports a 401
// as the words of its status. Transports fills it for every request whose context carries it,
// and leaves it alone otherwise. It holds no header, body or credential.
type Exchange struct {
	mu         sync.Mutex
	sent       bool
	status     int
	credential error
	timedOut   bool
	retryAfter time.Duration
	// scopeRequired is the provider asking for more access than the grant has, nil when no
	// answer did.
	scopeRequired *Outcome
}

type exchangeKey struct{}

// WithExchange is ctx carrying a new Exchange, and that Exchange.
func WithExchange(ctx context.Context) (context.Context, *Exchange) {
	e := &Exchange{}
	return context.WithValue(ctx, exchangeKey{}, e), e
}

// exchangeOf is the Exchange ctx carries, nil when it carries none.
func exchangeOf(ctx context.Context) *Exchange {
	e, _ := ctx.Value(exchangeKey{}).(*Exchange)
	return e
}

// Sent says a request left with a credential.
func (e *Exchange) Sent() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.sent
}

// Status is the provider's status for the last request that was answered, 0 when none was.
func (e *Exchange) Status() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.status
}

// RetryAfter is the wait the last answered request's 429 asked for, as the connection's scheme
// read its Retry-After (Scheme.Classify), zero when it was no 429 or said none.
func (e *Exchange) RetryAfter() time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.retryAfter
}

// CredentialError is why the Resolver gave the last request no credential, nil when it gave
// one.
func (e *Exchange) CredentialError() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.credential
}

// TimedOut says a request ended because the connection's client stopped waiting for it.
func (e *Exchange) TimedOut() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.timedOut
}

// ScopeRequired is the provider's ask for more access than the grant has: a scope_required
// Outcome with the scopes or the claims challenge to consent to (Scheme.Classify). It
// reports false when no answer asked.
func (e *Exchange) ScopeRequired() (Outcome, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.scopeRequired == nil {
		return Outcome{}, false
	}
	return *e.scopeRequired, true
}

// refused records that the Resolver gave a request no credential.
func (e *Exchange) refused(err error) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.credential = err
}

// sending records that a request is leaving with a credential.
func (e *Exchange) sending() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.sent = true
}

// answered records how a request that left ended: with the provider's status, or with err.
func (e *Exchange) answered(status int, err error) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err != nil {
		var timeout net.Error
		e.timedOut = e.timedOut || errors.Is(err, context.DeadlineExceeded) || errors.As(err, &timeout) && timeout.Timeout()
		return
	}
	e.status, e.retryAfter = status, 0
}

// limited records the wait the 429 that answered the last request asked for.
func (e *Exchange) limited(wait time.Duration) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.retryAfter = wait
}

// asked records that the provider asked for more access than the grant has.
func (e *Exchange) asked(outcome Outcome) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.scopeRequired = &outcome
}
