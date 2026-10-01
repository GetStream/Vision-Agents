package core

import (
	"context"
	"time"
)

// ConnectionRef names one connection within one tenant. It is all a caller passes to get a
// credential, so no caller can choose the scheme, the material or the store.
type ConnectionRef struct {
	CustomerID   string
	ConnectionID string
}

// Resolver is the only door to a credential. Every Source uses it; none opens the store or
// calls a token endpoint, which is also where a broker would plug in.
type Resolver interface {
	Resolve(ctx context.Context, ref ConnectionRef, need Need) (Credential, error)
	// Invalidate marks the connection as needing a reconnect and drops anything cached, so
	// a revoked grant stops being used before it next fails.
	Invalidate(ctx context.Context, ref ConnectionRef, why Outcome) error
}

// Need is what one call asks of a credential.
type Need struct {
	Audience string
	Scopes   []string
	// Deadline is the call's budget. The mint itself runs on a detached context, so a call
	// that gives up does not leave a refresh half done.
	Deadline time.Time
}

// Backend is the locked, revisioned storage behind the resolver.
//
// WithLocked loads the grant under a lock that holds across replicas and runs fn. fn may
// call checkpoint to persist the grant before a side effect it cannot take back, such as
// spending a rotating refresh token; returning changed persists the final state.
type Backend interface {
	WithLocked(ctx context.Context, ref ConnectionRef,
		fn func(g *Grant, checkpoint func() error) (changed bool, err error)) error
}

// Grant is the part of a connection that changes under the lock: the opened material and
// the state that moves with it. Revision advances with every new Material, and the
// material is sealed against it, so a stale write and a replayed blob both fail.
type Grant struct {
	Revision int
	// Status is connected, needs_reauthorization or disconnected.
	Status   string
	Material Material
	// ExpiresAt is when the current access expires; zero when it does not.
	ExpiresAt time.Time
	// LastError is shown to whoever has to reconnect, so it says what to do, not a stack.
	LastError string
}
