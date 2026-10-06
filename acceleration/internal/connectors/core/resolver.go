package core

import (
	"context"
	"time"
)

// ConnectionRef names one connection within one tenant. It is all a caller passes to get a
// credential, so no caller can choose the scheme, the stored credentials or the store.
type ConnectionRef struct {
	CustomerID   string
	ConnectionID string
}

// Resolver is the only door to a credential. Every ToolSource uses it; none opens the store or
// calls a token endpoint, which is also where a broker would plug in.
type Resolver interface {
	Resolve(ctx context.Context, ref ConnectionRef, req CredentialRequest) (AccessCredential, error)
	// Invalidate marks the connection as needing a reconnect and drops anything cached, so
	// a revoked grant stops being used before it next fails.
	Invalidate(ctx context.Context, ref ConnectionRef, why Outcome) error
}

// CredentialRequest is what one call asks of a credential.
type CredentialRequest struct {
	Audience string
	Scopes   []string
	// Deadline is the call's budget. Getting the access credential runs on a detached context, so a call
	// that gives up does not leave a refresh half done.
	Deadline time.Time
}

// CredentialStore is the locked, revisioned storage behind the resolver.
//
// Update loads the credential state under a lock that holds across replicas and runs fn.
// fn may call checkpoint to persist the state before a side effect it cannot take back, such as
// spending a rotating refresh token; returning changed persists the final state.
type CredentialStore interface {
	Update(ctx context.Context, ref ConnectionRef,
		fn func(state *CredentialState, checkpoint func() error) (changed bool, err error)) error
}

// CredentialState is the part of a connection that changes under the lock: the opened
// StoredCredentials and the state that moves with them. Revision advances with every new
// StoredCredentials, and they are sealed against it, so a stale write and a replayed blob both fail.
type CredentialState struct {
	Revision int
	// Status is connected, needs_reauthorization or disconnected.
	Status      string
	Credentials StoredCredentials
	// ExpiresAt is when the current access expires; zero when it does not.
	ExpiresAt time.Time
	// LastError is shown to whoever has to reconnect, so it says what to do, not a stack.
	LastError string
	// AccountID, Metadata and Scopes are what the consent that stored Credentials learned
	// (AccountInfo, from Scheme.Complete). A refresh keeps them. A reconnect compares
	// AccountID under the lock before it replaces anything, because a consent for another
	// account is another connection, not an update (architecture doc, one-way door 1).
	// AccountInfo.Unverified is not kept here.
	AccountID string
	Metadata  map[string]string
	Scopes    []string
}
