package core

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// Scheme is one way to acquire stored credentials, get an access credential from them, apply
// it and revoke them.
//
// There is one implementation per scheme, not per provider. A provider's quirks are
// ResolvedManifest fields; a provider that needs a new Scheme is a sign the manifest is missing a field.
type Scheme interface {
	Name() string
	// Begin starts acquisition. Interactive schemes return an authorize URL and the attempt
	// state to seal. Non-interactive schemes (api_key, client_credentials) return Done.
	Begin(ctx context.Context, in BeginInput) (BeginOutput, error)
	// Complete turns a callback or a supplied value into long-lived StoredCredentials and the public
	// values the manifest asked to capture.
	Complete(ctx context.Context, in CompleteInput) (StoredCredentials, AccountInfo, error)
	// Retrieve gets a short-lived AccessCredential from StoredCredentials, refreshing over
	// the network when it must, as aws.CredentialsProvider.Retrieve does. It may rotate
	// the StoredCredentials; the resolver persists what comes back under the lock. A
	// failed renewal returns an *OutcomeError and no StoredCredentials, and, while the old
	// access credential has not expired yet, that credential beside the error, so the call
	// can still go out. A renewal that cannot be taken back calls Checkpoint(ctx) first.
	Retrieve(ctx context.Context, stored StoredCredentials, m ResolvedManifest) (AccessCredential, StoredCredentials, error)
	// Wrap applies the credential to every outbound request: a header, a signature or a TLS
	// client certificate. base is the egress transport (egress.NewClient passes it), so
	// Wrap runs first and the egress check runs on the request Wrap produced, right before
	// the dial: it sees the URL that is dialed, and since it leaves the body alone a
	// signature made in Wrap still matches the wire.
	Wrap(base http.RoundTripper, c AccessCredential) http.RoundTripper
	// Classify maps a provider response to the one outcome the core acts on.
	Classify(resp *http.Response, body []byte, err error) Outcome
	// Revoke is best effort. A nil error is not proof the provider revoked anything.
	Revoke(ctx context.Context, stored StoredCredentials, m ResolvedManifest) error
}

// BeginInput is what a scheme needs to start acquiring a credential for one connection.
type BeginInput struct {
	Ref      ConnectionRef
	Manifest ResolvedManifest
	// RedirectURI is where an interactive scheme sends the browser back to.
	RedirectURI string
}

// BeginOutput is either an authorize URL with state to keep, or Done.
type BeginOutput struct {
	AuthorizeURL string
	// State is scheme-private (PKCE verifier, client registration). The core seals it into
	// the attempt and hands it back to Complete; the scheme never stores it itself.
	State json.RawMessage
	// Done means no interaction is needed and Complete can run at once.
	Done bool
}

// CompleteInput carries whatever arrived to finish an acquisition.
type CompleteInput struct {
	Ref      ConnectionRef
	Manifest ResolvedManifest
	// State is what Begin returned, unsealed.
	State json.RawMessage
	// Query is the full callback query, not only code and state, because capture rules
	// and the BeforeComplete hook read parameters a provider adds (realmId, hmac).
	Query url.Values
	// Supplied is what the developer gave a non-interactive scheme: an API key, a token,
	// a client secret. It is secret and goes into StoredCredentials, never into AccountInfo.
	Supplied map[string]string
}

// AccountInfo is what Complete learned that is public: it is stored on the connection,
// outside the sealed StoredCredentials, so it can be read and shown without the key.
type AccountInfo struct {
	// AccountID is the provider-side identity the manifest's identity rule produced. A
	// reconnect that yields a different one is a different account, not an update.
	AccountID string
	// Metadata holds the values capture rules named: instance_url, realm_id, team_id.
	Metadata map[string]string
	// Scopes are the scopes the provider says it granted.
	Scopes []string
	// Unverified names the Metadata values read from a callback query that a request with
	// the access token must confirm before they are trusted. When an identity part is
	// among them, AccountID is unverified too.
	Unverified []string
}

// StoredCredentials is scheme-private and sealed as one blob, bound to tenant, connection and
// revision. Only Scheme says what is in Payload, so a new scheme adds no column and
// re-shaping one scheme's payload never re-seals another's.
type StoredCredentials struct {
	Scheme  string          `json:"scheme"`
	Version int             `json:"version"`
	Payload json.RawMessage `json:"payload"`
}

// String keeps Payload, the long-lived secret, out of any %v or %s. JSON still carries it,
// because the marshaled bytes are what gets sealed.
func (s StoredCredentials) String() string {
	return "StoredCredentials{Scheme:" + s.Scheme + " Version:" + strconv.Itoa(s.Version) + " Payload:redacted}"
}

// GoString keeps Payload out of %#v, which ignores String.
func (s StoredCredentials) GoString() string {
	return s.String()
}

// LogValue keeps Payload out of slog, whose JSON handler would otherwise marshal it.
func (s StoredCredentials) LogValue() slog.Value {
	return slog.GroupValue(slog.String("scheme", s.Scheme), slog.Int("version", s.Version), slog.String("payload", "redacted"))
}

// AccessCredential is what one request needs. It never reaches a log or the model: the secret is
// unexported, left out of JSON, and redacted by String and GoString.
type AccessCredential struct {
	Scheme    string
	ExpiresAt time.Time
	secret    json.RawMessage
}

// NewAccessCredential is how a scheme in its own package builds an AccessCredential at all, since the
// secret field is unexported.
func NewAccessCredential(scheme string, expiresAt time.Time, secret json.RawMessage) AccessCredential {
	return AccessCredential{Scheme: scheme, ExpiresAt: expiresAt, secret: secret}
}

// Secret is for the scheme that issued the credential, in Wrap. Nothing else reads it.
func (c AccessCredential) Secret() json.RawMessage {
	return c.secret
}

// String keeps the secret out of any %v or %s, which is how most log lines are made.
func (c AccessCredential) String() string {
	return "AccessCredential{Scheme:" + c.Scheme + " ExpiresAt:" + c.ExpiresAt.String() + " secret:redacted}"
}

// GoString keeps the secret out of %#v, which ignores String.
func (c AccessCredential) GoString() string {
	return c.String()
}

// OutcomeKind is one of the few things the core does after a provider answers. It is a
// string so the zero value is no outcome at all, not OK.
type OutcomeKind string

const (
	// OutcomeOK means the call worked.
	OutcomeOK OutcomeKind = "ok"
	// OutcomeInvalidGrant means the provider rejected the grant; only a reconnect helps.
	OutcomeInvalidGrant OutcomeKind = "invalid_grant"
	// OutcomeTransient means the provider failed in a way a later attempt may not.
	OutcomeTransient OutcomeKind = "transient"
	// OutcomeUncertain means the request may have taken effect but the answer was lost,
	// so a rotated refresh token must not be replayed.
	OutcomeUncertain OutcomeKind = "uncertain"
	// OutcomeScopeRequired means the grant lacks a scope or a claims challenge was issued.
	OutcomeScopeRequired OutcomeKind = "scope_required"
	// OutcomeRateLimited means the provider asked to slow down.
	OutcomeRateLimited OutcomeKind = "rate_limited"
)

// Outcome is a classified provider response.
type Outcome struct {
	Kind OutcomeKind
	// Scopes is, for ScopeRequired, the union to ask for.
	Scopes []string
	// Claims is, for ScopeRequired, a claims challenge to pass back on consent.
	Claims     string
	RetryAfter time.Duration
}

// OutcomeError is a failed AccessCredential or Revoke with the outcome it was classified as, so the
// resolver acts on Outcome (errors.As) without knowing the scheme. Err says what happened
// and, like every error a scheme returns, carries no secret.
type OutcomeError struct {
	Outcome Outcome
	Err     error
}

func (e *OutcomeError) Error() string {
	return string(e.Outcome.Kind) + ": " + e.Err.Error()
}

func (e *OutcomeError) Unwrap() error {
	return e.Err
}
