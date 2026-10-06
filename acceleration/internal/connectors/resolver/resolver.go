// Package resolver is the core.Resolver the router uses: it reads a connection's row, hands
// out a cached access credential while the row still says it may, and otherwise gets one
// through the connection's scheme under the credential store's lock, with the checkpoint
// before any refresh, and moves the connection's status by what the scheme answered.
package resolver

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// retrieveTimeout bounds one Retrieve, which runs detached from the caller's context so a
// call that gives up does not leave a refresh half done (architecture doc, one-way door 4).
// It covers the two token requests a refresh can make, the first and one grace retry
// (oauth2code refresh), at the router's connectorHTTPTimeout of 10 s each (cmd/router, the
// prototype's), with room for the client lookup and the checkpoint write
// (store.credentialDetachedTimeout, 2 s). Unverified, not measured.
const retrieveTimeout = 30 * time.Second

// maxAge is how long a cached access credential is handed out before the scheme is asked
// again, even when the connection's row has not moved. The row itself is read on every call,
// so this bounds only what the row does not show: a credential the scheme would now renew,
// and stored credentials under an old key that the credential store rewraps on its next use.
// Half of oauth2code's default refresh margin (1 minute, the prototype's runtime.go:78 at
// cf62af0d), so a token cached just outside that margin is renewed at most 30 s late and
// before it expires. Unverified, not measured (spike 4 measures the fast path, not this).
const maxAge = 30 * time.Second

// What LastError says after each outcome. core.CredentialState.LastError is shown to whoever
// has to reconnect, so each says what to do. The wording follows the prototype's
// (ResolveCredentials in internal/connectors/runtime.go:98-115 at cf62af0d), without the word
// OAuth, since any scheme can end up here.
const (
	// lostRefresh is what the checkpoint commits before a refresh, and what stays when its
	// answer is lost: the provider may have rotated the refresh token, and only a reconnect is
	// sure to work (runtime.go:98).
	lostRefresh = "A credential refresh did not finish durably; reconnect the account"
	// rejectedGrant is the provider refusing the grant (runtime.go:106).
	rejectedGrant = "The provider rejected the grant; reconnect the account"
	// missingScope is the provider asking for access the grant does not have. New wording:
	// the prototype had no such outcome; a reconnect with the wider scopes is what helps.
	missingScope = "The provider asks for access the grant does not have; reconnect the account to grant it"
	// temporarilyUnavailable is a refresh that failed in a way a later call may not; the
	// connection stays connected (runtime.go:115).
	temporarilyUnavailable = "The provider could not renew the credential just now; the next call tries again"
)

// ErrNotConnected says the connection's status is not connected: it is pending a consent, it
// needs a reconnect, or it was disconnected. Only the person who owns it can fix that.
var ErrNotConnected = errors.New("resolver: the connection is not connected")

// ErrTemporarilyUnavailable says the provider failed to renew an expired credential in a way
// a later call may not. The connection stays connected; the *core.OutcomeError it wraps
// carries the provider's Retry-After, if it sent one.
var ErrTemporarilyUnavailable = errors.New("resolver: the provider could not renew the credential just now")

// Config is what a Resolver reads from.
type Config struct {
	// Store holds the connections and their definitions.
	Store *store.Store
	// Credentials is the locked, revisioned credential store, pgsealed in the router.
	Credentials core.CredentialStore
	// Schemes are the registered schemes by name, as a connection's auth_scheme names them.
	Schemes map[string]core.Scheme
	// Now is the clock the cache is judged by; nil is time.Now.
	Now func() time.Time
}

// Resolver implements core.Resolver.
type Resolver struct {
	store       *store.Store
	credentials core.CredentialStore
	schemes     map[string]core.Scheme
	now         func() time.Time

	mu    sync.Mutex
	cache map[core.ConnectionRef]entry
	// swept is when expired entries were last dropped, so connections nobody resolves any
	// more do not keep theirs.
	swept time.Time
}

var _ core.Resolver = (*Resolver)(nil)

// entry is one access credential and the revision of the stored credentials it came from.
type entry struct {
	revision   int
	credential core.AccessCredential
	fetched    time.Time
	// renewed says the Retrieve that fetched it renewed the stored credentials.
	renewed bool
}

// New is a Resolver over cfg.
func New(cfg Config) (*Resolver, error) {
	if cfg.Store == nil || cfg.Credentials == nil {
		return nil, stack.Wrap(errors.New("resolver: a store and a credential store are required"))
	}
	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	return &Resolver{store: cfg.Store, credentials: cfg.Credentials, schemes: cfg.Schemes, now: now,
		cache: map[core.ConnectionRef]entry{}}, nil
}

// Resolve returns an access credential for ref.
//
// It reads the connection's row on every call, without the lock: a deleted connection is
// store.ErrNoConnectorConnection at once, on every router, whatever is cached. A connected
// row at the revision a cached credential came from gets that credential while it is younger
// than maxAge and outlives req.Deadline (or was renewed, see cached). Anything else runs the
// scheme's Retrieve under the credential store's lock, asking for a credential that outlives
// req.Deadline, on a context of its own: a caller whose ctx ends during a refresh gets ctx's
// error at once, and the refresh finishes and is committed without it.
func (r *Resolver) Resolve(ctx context.Context, ref core.ConnectionRef, req core.CredentialRequest) (core.AccessCredential, error) {
	connection, err := r.store.ConnectorConnection(ctx, ref.CustomerID, ref.ConnectionID)
	if err != nil {
		r.drop(ref)
		return core.AccessCredential{}, err
	}
	if connection.Status == store.ConnectionConnected {
		if credential, found := r.cached(ref, connection.Revision, req.Deadline); found {
			return credential, nil
		}
	}

	type result struct {
		credential core.AccessCredential
		err        error
	}
	done := make(chan result, 1)
	go func() {
		credential, err := r.retrieve(ctx, ref, connection, req.Deadline)
		done <- result{credential, err}
	}()
	select {
	case got := <-done:
		return got.credential, got.err
	case <-ctx.Done():
		return core.AccessCredential{}, stack.Wrap(context.Cause(ctx))
	}
}

// Invalidate moves a connected connection to needs_reauthorization and drops its cached
// credential, for a provider that refused the credential on a call (why is invalid_grant) or
// asked for more access than it grants (scope_required). Any other outcome ends no grant and
// is refused. A connection that is not connected keeps its status.
func (r *Resolver) Invalidate(ctx context.Context, ref core.ConnectionRef, why core.Outcome) error {
	var lastError string
	switch why.Kind {
	case core.OutcomeInvalidGrant:
		lastError = rejectedGrant
	case core.OutcomeScopeRequired:
		lastError = missingScope
	default:
		return stack.Wrap(fmt.Errorf("resolver: %q ends no grant; only %q and %q invalidate a connection",
			why.Kind, core.OutcomeInvalidGrant, core.OutcomeScopeRequired))
	}
	r.drop(ref)
	return stack.Wrap(r.credentials.Update(ctx, ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		if state.Status != store.ConnectionConnected {
			return false, nil
		}
		state.Status, state.LastError = store.ConnectionNeedsReauthorization, lastError
		return true, nil
	}))
}

// retrieve gets an access credential that works until validUntil through the connection's
// scheme under the credential store's lock and commits what it learned. ctx only bounds the
// wait for the lock and the reads; the scheme runs on a detached context.
func (r *Resolver) retrieve(ctx context.Context, ref core.ConnectionRef, connection store.ConnectorConnection, validUntil time.Time) (core.AccessCredential, error) {
	scheme, found := r.schemes[connection.AuthScheme]
	if !found {
		return core.AccessCredential{}, stack.Wrap(fmt.Errorf("%w: %q", store.ErrUnregisteredScheme, connection.AuthScheme))
	}
	// A connection's definition revision, inputs and scheme are never rewritten
	// (store.credentialColumns), so they are read outside the lock. Its metadata can change
	// with a reconnect, so the manifest is resolved inside it.
	definition, err := r.store.ConnectorDefinition(ctx, ref.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return core.AccessCredential{}, err
	}

	var (
		credential core.AccessCredential
		failure    error
		cache      bool
		before     core.CredentialState
		committed  *core.CredentialState
	)
	err = r.credentials.Update(ctx, ref, func(state *core.CredentialState, checkpoint func() error) (bool, error) {
		committed = state
		if state.Status != store.ConnectionConnected {
			failure = stack.Wrap(fmt.Errorf("%w: it is %s", ErrNotConnected, state.Status))
			return false, nil
		}
		manifest, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, state.Metadata)
		if err != nil {
			return false, err
		}
		before = *state
		checkpointed := false
		detached, cancel := context.WithTimeout(context.WithoutCancel(ctx), retrieveTimeout)
		defer cancel()
		got, next, err := scheme.Retrieve(detached, state.Credentials, manifest, core.RetrieveOptions{
			ValidUntil: validUntil,
			Checkpoint: func() error {
				state.Status, state.LastError = store.ConnectionNeedsReauthorization, lostRefresh
				if err := checkpoint(); err != nil {
					return err
				}
				checkpointed = true
				return nil
			},
		})
		var outcome *core.OutcomeError
		switch {
		case err == nil:
			state.Credentials = next
			state.Status, state.LastError = store.ConnectionConnected, ""
			// The credential store keeps microseconds, so the expiry is compared as stored.
			state.ExpiresAt = got.ExpiresAt.UTC().Truncate(time.Microsecond)
			credential, cache = got, true
		case errors.As(err, &outcome):
			state.Status, state.LastError, failure = statusAfter(outcome, err)
			// A renewal that failed before the old access credential expired hands that one
			// back beside the error, for this call only (core.Scheme.Retrieve).
			if got.Scheme != "" {
				credential, failure = got, nil
			}
		default:
			// Unreadable stored credentials, a missing client or a failed checkpoint: nothing
			// the provider said. A checkpoint that committed stays.
			return false, err
		}
		return checkpointed || changed(before, *state), nil
	})
	if err != nil {
		return core.AccessCredential{}, stack.Wrap(err)
	}
	if failure != nil {
		return core.AccessCredential{}, failure
	}
	// The credential store leaves state at the revision it committed (core.CredentialStore),
	// so the cache never numbers revisions itself.
	if cache {
		r.put(ref, entry{revision: committed.Revision, credential: credential, fetched: r.now(),
			renewed: committed.Revision != before.Revision})
	}
	return credential, nil
}

// statusAfter is the status and LastError a failed renewal leaves, and the error a caller
// gets when no credential came back with it.
func statusAfter(outcome *core.OutcomeError, err error) (status, lastError string, failure error) {
	switch outcome.Outcome.Kind {
	case core.OutcomeInvalidGrant:
		return store.ConnectionNeedsReauthorization, rejectedGrant, stack.Wrap(fmt.Errorf("%w: %w", ErrNotConnected, err))
	case core.OutcomeScopeRequired:
		return store.ConnectionNeedsReauthorization, missingScope, stack.Wrap(fmt.Errorf("%w: %w", ErrNotConnected, err))
	case core.OutcomeUncertain:
		// The refresh may have rotated the token at the provider, so it is never sent again.
		return store.ConnectionNeedsReauthorization, lostRefresh, stack.Wrap(fmt.Errorf("%w: %w", ErrNotConnected, err))
	default:
		// Transient and RateLimited changed nothing at the provider, and neither does an
		// error that names no outcome the core acts on.
		return store.ConnectionConnected, temporarilyUnavailable, stack.Wrap(fmt.Errorf("%w: %w", ErrTemporarilyUnavailable, err))
	}
}

// cached is the credential cached for ref at revision, while it is younger than maxAge and
// does not expire before deadline (or now, when the call has none). A credential this
// router renewed within maxAge is handed out while it has not expired, even when it expires
// before deadline: it is as long as the provider issues them, so renewing again would give
// one no longer, and asking for that on every call would be a refresh per call.
func (r *Resolver) cached(ref core.ConnectionRef, revision int, deadline time.Time) (core.AccessCredential, bool) {
	now := r.now()
	r.mu.Lock()
	defer r.mu.Unlock()
	cached, found := r.cache[ref]
	if !found || cached.revision != revision || !now.Before(cached.fetched.Add(maxAge)) {
		return core.AccessCredential{}, false
	}
	if deadline.Before(now) {
		deadline = now
	}
	expires := cached.credential.ExpiresAt
	if expires.IsZero() || deadline.Before(expires) || (cached.renewed && now.Before(expires)) {
		return cached.credential, true
	}
	return core.AccessCredential{}, false
}

// put caches e for ref, replacing what an earlier revision left, and now and then drops
// the entries that are past maxAge.
func (r *Resolver) put(ref core.ConnectionRef, e entry) {
	now := r.now()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cache[ref] = e
	if now.Sub(r.swept) < maxAge {
		return
	}
	for key, cached := range r.cache {
		if !now.Before(cached.fetched.Add(maxAge)) {
			delete(r.cache, key)
		}
	}
	r.swept = now
}

func (r *Resolver) drop(ref core.ConnectionRef) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.cache, ref)
}

// changed says whether Retrieve moved anything the credential store keeps, so a call that
// changed nothing writes nothing.
func changed(before, after core.CredentialState) bool {
	return before.Status != after.Status || before.LastError != after.LastError ||
		!before.ExpiresAt.Equal(after.ExpiresAt) || !sameCredentials(before.Credentials, after.Credentials)
}

func sameCredentials(a, b core.StoredCredentials) bool {
	return a.Scheme == b.Scheme && a.Version == b.Version && bytes.Equal(a.Payload, b.Payload)
}
