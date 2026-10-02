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

// Scheme is one way to acquire, mint, apply and revoke a credential.
//
// There is one implementation per scheme, not per provider. A provider's quirks are Profile
// fields; a provider that needs a new Scheme is a sign the manifest is missing a field.
type Scheme interface {
	Name() string
	// Begin starts acquisition. Interactive schemes return an authorize URL and the attempt
	// state to seal. Non-interactive schemes (api_key, client_credentials) return Done.
	Begin(ctx context.Context, in BeginInput) (BeginOutput, error)
	// Complete turns a callback or a supplied value into long-lived Material and the public
	// values the manifest asked to capture.
	Complete(ctx context.Context, in CompleteInput) (Material, Captured, error)
	// Mint produces a short-lived Credential from Material. It may rotate Material; the
	// resolver persists what comes back under the lock.
	Mint(ctx context.Context, m Material, p Profile) (Credential, Material, error)
	// Wrap applies the credential to every outbound request: a header, a signature or a TLS
	// client certificate. base is the egress transport (egress.NewClient passes it), so
	// Wrap runs first and the egress check runs on the request Wrap produced, right before
	// the dial: it sees the URL that is dialed, and since it leaves the body alone a
	// signature made in Wrap still matches the wire.
	Wrap(base http.RoundTripper, c Credential) http.RoundTripper
	// Classify maps a provider response to the one outcome the core acts on.
	Classify(resp *http.Response, body []byte, err error) Outcome
	// Revoke is best effort. A nil error is not proof the provider revoked anything.
	Revoke(ctx context.Context, m Material, p Profile) error
}

// BeginInput is what a scheme needs to start acquiring a credential for one connection.
type BeginInput struct {
	Ref     ConnectionRef
	Profile Profile
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
	Ref     ConnectionRef
	Profile Profile
	// State is what Begin returned, unsealed.
	State json.RawMessage
	// Query is the full callback query, not only code and state, because capture rules
	// and the BeforeComplete hook read parameters a provider adds (realmId, hmac).
	Query url.Values
	// Supplied is what the developer gave a non-interactive scheme: an API key, a token,
	// a client secret. It is secret and goes into Material, never into Captured.
	Supplied map[string]string
}

// Captured is what Complete learned that is public: it is stored on the connection,
// outside the sealed Material, so it can be read and shown without the key.
type Captured struct {
	// AccountID is the provider-side identity the manifest's identity rule produced. A
	// reconnect that yields a different one is a different account, not an update.
	AccountID string
	// Metadata holds the values capture rules named: instance_url, realm_id, team_id.
	Metadata map[string]string
	// Scopes are the scopes the provider says it granted.
	Scopes []string
	// Unverified names the Metadata values read from a callback query that a request with
	// the minted token must confirm before they are trusted. When an identity part is
	// among them, AccountID is unverified too.
	Unverified []string
}

// Material is scheme-private and sealed as one blob, bound to tenant, connection and
// revision. Only Scheme says what is in Payload, so a new scheme adds no column and
// re-shaping one scheme's payload never re-seals another's.
type Material struct {
	Scheme  string          `json:"scheme"`
	Version int             `json:"version"`
	Payload json.RawMessage `json:"payload"`
}

// String keeps Payload, the long-lived secret, out of any %v or %s. JSON still carries it,
// because the marshaled bytes are what gets sealed.
func (m Material) String() string {
	return "Material{Scheme:" + m.Scheme + " Version:" + strconv.Itoa(m.Version) + " Payload:redacted}"
}

// GoString keeps Payload out of %#v, which ignores String.
func (m Material) GoString() string {
	return m.String()
}

// LogValue keeps Payload out of slog, whose JSON handler would otherwise marshal it.
func (m Material) LogValue() slog.Value {
	return slog.GroupValue(slog.String("scheme", m.Scheme), slog.Int("version", m.Version), slog.String("payload", "redacted"))
}

// Credential is what one request needs. It never reaches a log or the model: the secret is
// unexported, left out of JSON, and redacted by String and GoString.
type Credential struct {
	Scheme    string
	ExpiresAt time.Time
	secret    json.RawMessage
}

// NewCredential is how a scheme in its own package builds a Credential at all, since the
// secret field is unexported.
func NewCredential(scheme string, expiresAt time.Time, secret json.RawMessage) Credential {
	return Credential{Scheme: scheme, ExpiresAt: expiresAt, secret: secret}
}

// Secret is for the scheme that minted the credential, in Wrap. Nothing else reads it.
func (c Credential) Secret() json.RawMessage {
	return c.secret
}

// String keeps the secret out of any %v or %s, which is how most log lines are made.
func (c Credential) String() string {
	return "Credential{Scheme:" + c.Scheme + " ExpiresAt:" + c.ExpiresAt.String() + " secret:redacted}"
}

// GoString keeps the secret out of %#v, which ignores String.
func (c Credential) GoString() string {
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

// OutcomeError is a failed Mint or Revoke with the outcome it was classified as, so the
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
