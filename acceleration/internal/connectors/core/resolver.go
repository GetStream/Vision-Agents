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
	// a revoked grant stops being used before it next fails. rejected is the credential the
	// provider refused, as Resolve returned it. A refusal of a credential that had expired,
	// or whose Revision the stored credentials have moved past (another router renewed
	// them), says nothing about the grant: then only the cache is dropped.
	Invalidate(ctx context.Context, ref ConnectionRef, rejected AccessCredential, why Outcome) error
	// Revoke marks a connected connection as needing a reconnect and drops anything cached,
	// because the provider said the grant ended (a verified Signal of kind why). Unlike
	// Invalidate it names no credential: the provider ended the grant itself, whatever
	// revision the stored credentials are at, so the status moves under the lock whichever
	// router renewed them last. endedAt is when the provider says the grant ended
	// (Signal.At), zero when it does not say: a connection a consent connected after it
	// holds a newer grant than the one the signal is about, and stays connected.
	Revoke(ctx context.Context, ref ConnectionRef, why SignalKind, endedAt time.Time) error
}

// CredentialRequest is what one call asks of a credential.
type CredentialRequest struct {
	Audience string
	Scopes   []string
	// Deadline is the call's budget. The access credential handed out still works then: the
	// resolver asks the scheme for that (RetrieveOptions.ValidUntil). Getting it runs on a
	// detached context, so a call that gives up does not leave a refresh half done.
	Deadline time.Time
	// Refused is the access credential the provider just refused on this call, as Resolve
	// returned it, or nil. The resolver then hands out none from its cache and, while the
	// stored credentials are still at Refused.Revision, asks the scheme to renew whatever
	// the expiry says (RetrieveOptions.Refused). A renewal the provider refuses moves the
	// connection as any failed renewal does; a scheme that cannot renew hands back the same
	// credential, and the caller then calls Invalidate.
	Refused *AccessCredential
}

// CredentialStore is the locked, revisioned storage behind the resolver.
//
// Update loads the credential state under a lock that holds across replicas and runs fn.
// fn may call checkpoint to persist the state before a side effect it cannot take back, such as
// spending a rotating refresh token; returning changed persists the final state. Each commit,
// the checkpoint's and the final one, leaves state.Revision at the revision it committed, so
// whoever ran Update reads the committed revision from state once Update returns.
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
	// ConnectedAt is when a consent last stored Credentials: when the grant they belong to
	// began. A refresh keeps it. Zero for a connection connected before it was kept.
	ConnectedAt time.Time
	// DefinitionRevision is the manifest revision the connection reads: the one the consent
	// that stored Credentials ran on, which sets it. A refresh keeps it.
	DefinitionRevision int
}
