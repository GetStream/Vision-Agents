// Package oauth2code is the oauth2_code scheme: the OAuth 2.0 authorization code grant
// with PKCE (RFC 6749 section 4.1, RFC 7636), for every provider a manifest describes.
//
// Begin discovers the authorization server when the manifest does not pin it, picks or
// registers a client, and builds the authorize URL. Complete checks the callback against
// the state Begin returned, redeems the code and applies the manifest's capture and
// identity rules. AccessCredential renews the access token by the manifest's refresh policy,
// Wrap puts it on a request, Classify says what a provider's answer means, and Revoke ends
// the grant at the provider. Nothing here knows a provider: what differs between them is a
// core.ResolvedManifest field.
package oauth2code

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
)

// Name is the registry name manifests list under schemes.
const Name = "oauth2_code"

const (
	// randomBytes is the size of the state and of the PKCE verifier: RFC 7636 section 4.1
	// recommends a 32-octet random verifier, base64url-encoded into 43 characters. 256
	// random bits also put a guessed state far below RFC 6749 section 10.10's 2^-160.
	randomBytes = 32
	// attemptTTL is how long a consent may take. It is T17's attempt lifetime (AI-816
	// subtasks, T17) and RFC 6749 section 4.1.2's recommended maximum code lifetime, so no
	// code is still redeemable once the attempt it belongs to is over.
	attemptTTL = 10 * time.Minute
	// payloadVersion is the shape of storedPayload. A new shape is a new version, so
	// a sealed blob is always read as what it was written as.
	payloadVersion = 1
)

// Errors Complete returns for a callback it refuses. Each one means the code, if any, was
// not sent to the token endpoint.
var (
	// ErrUnknownState is a callback whose state is not the one Begin issued: a forged or
	// crossed callback (RFC 6749 section 10.12).
	ErrUnknownState = errors.New("oauth2code: callback state does not match the attempt")
	// ErrReplayedState is a callback for an attempt this scheme already completed. A second
	// redemption of the code would make the server revoke the grant (RFC 6749 section
	// 4.1.2), so it never leaves the router.
	ErrReplayedState = errors.New("oauth2code: callback state was already used")
	// ErrExpiredState is a callback after attemptTTL.
	ErrExpiredState = errors.New("oauth2code: attempt expired")
	// ErrIssuerMismatch is a callback naming another issuer than the one the request went
	// to: a mix-up (RFC 9207 section 2.4, RFC 9700 section 4.4).
	ErrIssuerMismatch = errors.New("oauth2code: callback iss does not match the authorization server")
	// ErrIssuerMissing is a callback without iss from a server whose metadata says it sends
	// one (RFC 9207 section 2.4).
	ErrIssuerMissing = errors.New("oauth2code: callback has no iss, and the authorization server sends one")
	// ErrNoClient means no client the manifest's client.registration allows is available.
	ErrNoClient = errors.New("oauth2code: no OAuth client available for this connector")
)

// AuthorizationError is the error a provider sent back to the callback (RFC 6749 section
// 4.1.2.1), such as access_denied when the user declined.
type AuthorizationError struct {
	Code        string
	Description string
}

func (e *AuthorizationError) Error() string {
	if e.Description == "" {
		return "oauth2code: authorization refused: " + e.Code
	}
	return "oauth2code: authorization refused: " + e.Code + ": " + e.Description
}

// Config is what the scheme depends on. The caller chooses the client every request leaves
// through; the one default that resolves names is PublicEndpoint's, egress's own check.
type Config struct {
	// HTTP carries every outbound request: discovery, registration and the code exchange.
	// In the router it is egress.NewClient(timeout, nil), so each one is checked against
	// the egress policy at dial; tests pass the fake provider's Client.
	HTTP *http.Client
	// Clients finds a preregistered client, the operator's or a customer's. Nil means none
	// is ever found, so only cimd and dcr can supply one.
	Clients ClientLookup
	// ClientMetadataURL is the router's client_id under CIMD: the https URL its client
	// metadata document is served at (ClientMetadataDocument). Empty turns CIMD off.
	ClientMetadataURL string
	// Now is the clock attempts expire by; nil is time.Now. Tests move it.
	Now func() time.Time
	// PublicEndpoint checks that an authorization server endpoint, with its query removed,
	// is a public https URL; nil is egress.ValidatePublicHTTPSURL. The authorize URL goes to
	// the browser and never through the egress client, so this is the only check it gets.
	// Tests that run against a loopback fake pass one that lets the fake's host through.
	PublicEndpoint func(ctx context.Context, raw string) error
	// Logger gets AccessCredential's warning that a connection is about to need a reconnect; nil is
	// slog.Default(). Nothing logged names a token.
	Logger *slog.Logger
}

// Scheme is the oauth2_code scheme. It is safe for concurrent use.
type Scheme struct {
	cfg Config

	mu sync.Mutex
	// spent holds the state of every attempt Complete took, until it expires. The one-use
	// attempt the core seals is the guard across replicas; this one stops a replay within
	// one process before its code reaches the provider.
	spent map[string]time.Time
}

var _ core.Scheme = (*Scheme)(nil)

// New checks cfg and returns the scheme.
func New(cfg Config) (*Scheme, error) {
	if cfg.HTTP == nil {
		return nil, errors.New("oauth2code: Config.HTTP is required")
	}
	if cfg.ClientMetadataURL != "" {
		if err := checkClientIdentifierURL(cfg.ClientMetadataURL); err != nil {
			return nil, err
		}
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.PublicEndpoint == nil {
		cfg.PublicEndpoint = egress.ValidatePublicHTTPSURL
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Scheme{cfg: cfg, spent: map[string]time.Time{}}, nil
}

// Name is Name.
func (s *Scheme) Name() string {
	return Name
}

// attempt is the State Begin returns and Complete reads. The core seals it, so it may carry
// the PKCE verifier and a registered client's secret.
type attempt struct {
	State string `json:"state"`
	// Verifier is the PKCE code_verifier (RFC 7636 section 4.1).
	Verifier string `json:"code_verifier"`
	// RedirectURI is sent again with the code, which RFC 6749 section 4.1.3 requires to be
	// identical to the authorize request's.
	RedirectURI   string    `json:"redirect_uri"`
	Client        client    `json:"client"`
	Issuer        string    `json:"issuer,omitempty"`
	RequireIssuer bool      `json:"require_iss,omitempty"`
	TokenEndpoint string    `json:"token_endpoint"`
	Revocation    string    `json:"revocation_endpoint,omitempty"`
	Resource      string    `json:"resource,omitempty"`
	Scopes        []string  `json:"scopes,omitempty"`
	ExpiresAt     time.Time `json:"expires_at"`
}

// storedPayload is the sealed payload of a connection's StoredCredentials: everything a
// refresh and a revocation need, so neither has to discover the server again.
type storedPayload struct {
	// Ref is the connection Complete ran for. AccessCredential and Revoke look a preregistered
	// client's secret up by it, as Complete did, since core.Scheme hands them no
	// ConnectionRef. The core seals StoredCredentials bound to that same connection, so it
	// cannot name another.
	Ref                core.ConnectionRef `json:"ref"`
	Client             client             `json:"client"`
	TokenEndpoint      string             `json:"token_endpoint"`
	RevocationEndpoint string             `json:"revocation_endpoint,omitempty"`
	Issuer             string             `json:"issuer,omitempty"`
	Resource           string             `json:"resource,omitempty"`
	AccessToken        string             `json:"access_token"`
	TokenType          string             `json:"token_type,omitempty"`
	RefreshToken       string             `json:"refresh_token,omitempty"`
	ExpiresAt          time.Time          `json:"expires_at,omitzero"`
	// RefreshExpiresAt is when RefreshToken dies by the manifest's refresh.refresh_ttl,
	// zero when the manifest does not say.
	RefreshExpiresAt time.Time `json:"refresh_expires_at,omitzero"`
	Scopes           []string  `json:"scopes,omitempty"`
}

// Begin discovers what the manifest leaves out, picks a client and returns the authorize
// URL. State is never Done: a person always has to consent.
func (s *Scheme) Begin(ctx context.Context, in core.BeginInput) (core.BeginOutput, error) {
	if in.RedirectURI == "" {
		return core.BeginOutput{}, errors.New("oauth2code: BeginInput.RedirectURI is required")
	}
	server, err := s.discover(ctx, in.Manifest)
	if err != nil {
		return core.BeginOutput{}, err
	}
	if err := server.checkPKCE(); err != nil {
		return core.BeginOutput{}, err
	}
	c, err := s.pickClient(ctx, in.Ref, in.Manifest, server, in.RedirectURI)
	if err != nil {
		return core.BeginOutput{}, err
	}
	state, err := random()
	if err != nil {
		return core.BeginOutput{}, err
	}
	verifier, err := random()
	if err != nil {
		return core.BeginOutput{}, err
	}

	authorize, err := url.Parse(server.Authorize)
	if err != nil {
		return core.BeginOutput{}, fmt.Errorf("oauth2code: authorize endpoint: %w", err)
	}
	// RFC 6749 section 3.1: a query the endpoint already has «MUST be retained».
	query := authorize.Query()
	// The manifest's extra parameters go first, so none of them can replace a protocol
	// parameter set below.
	for _, key := range slices.Sorted(maps.Keys(in.Manifest.AuthorizeParams)) {
		query.Set(key, in.Manifest.AuthorizeParams[key])
	}
	// RFC 6749 section 4.1.1.
	query.Set("response_type", "code")
	query.Set("client_id", c.ID)
	query.Set("redirect_uri", in.RedirectURI)
	query.Set("state", state)
	// RFC 7636 section 4.2 and 4.3. S256 only: RFC 9700 section 2.1.1, «Currently, S256 is
	// the only such method» that does not expose the verifier.
	digest := sha256.Sum256([]byte(verifier))
	query.Set("code_challenge", base64.RawURLEncoding.EncodeToString(digest[:]))
	query.Set("code_challenge_method", "S256")
	if server.Resource != "" {
		// RFC 8707 section 2.1.
		query.Set("resource", server.Resource)
	}
	if scopes := in.Manifest.Scopes.List; len(scopes) > 0 {
		query.Set("scope", strings.Join(scopes, separator(in.Manifest)))
	}
	authorize.RawQuery = query.Encode()

	raw, err := json.Marshal(attempt{
		State:         state,
		Verifier:      verifier,
		RedirectURI:   in.RedirectURI,
		Client:        c,
		Issuer:        server.Issuer,
		RequireIssuer: server.IssParameter,
		TokenEndpoint: server.Token,
		Revocation:    server.Revocation,
		Resource:      server.Resource,
		Scopes:        slices.Clone(in.Manifest.Scopes.List),
		ExpiresAt:     s.cfg.Now().Add(attemptTTL),
	})
	if err != nil {
		return core.BeginOutput{}, err
	}
	return core.BeginOutput{AuthorizeURL: authorize.String(), State: raw}, nil
}

// Complete checks the callback against the attempt, redeems the code and applies the
// resolved manifest's capture and identity rules to the callback and the token response.
func (s *Scheme) Complete(ctx context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	var a attempt
	if err := json.Unmarshal(in.State, &a); err != nil || a.State == "" {
		return core.StoredCredentials{}, core.AccountInfo{}, ErrUnknownState
	}
	now := s.cfg.Now()
	if !now.Before(a.ExpiresAt) {
		return core.StoredCredentials{}, core.AccountInfo{}, ErrExpiredState
	}
	// RFC 6749 section 10.12: the callback must carry the state this attempt issued.
	// Exactly one value, compared in constant time.
	got := in.Query["state"]
	if len(got) != 1 || subtle.ConstantTimeCompare([]byte(got[0]), []byte(a.State)) != 1 {
		return core.StoredCredentials{}, core.AccountInfo{}, ErrUnknownState
	}
	if !s.spend(a.State, a.ExpiresAt, now) {
		return core.StoredCredentials{}, core.AccountInfo{}, ErrReplayedState
	}
	// RFC 9207 section 2.4, for error responses too: until iss is checked an error may
	// come from another server.
	switch iss := in.Query["iss"]; {
	case len(iss) > 1:
		return core.StoredCredentials{}, core.AccountInfo{}, ErrIssuerMismatch
	case len(iss) == 1 && iss[0] != a.Issuer:
		return core.StoredCredentials{}, core.AccountInfo{}, ErrIssuerMismatch
	case len(iss) == 0 && a.RequireIssuer:
		return core.StoredCredentials{}, core.AccountInfo{}, ErrIssuerMissing
	}
	if code := in.Query.Get("error"); code != "" {
		return core.StoredCredentials{}, core.AccountInfo{}, &AuthorizationError{Code: code, Description: in.Query.Get("error_description")}
	}
	code := in.Query["code"]
	if len(code) != 1 || code[0] == "" {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: callback has no code")
	}

	c, err := s.clientSecret(ctx, in.Ref, in.Manifest, a.Client)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	token, raw, err := s.exchange(ctx, in.Manifest, a, c, code[0])
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	if readsIDToken(in.Manifest) {
		if err := checkIDToken(token, a.Issuer, a.Client.ID, now); err != nil {
			return core.StoredCredentials{}, core.AccountInfo{}, err
		}
	}
	account, err := in.Manifest.Apply(in.Query, raw)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, fmt.Errorf("oauth2code: %w", err)
	}
	account.Scopes = token.scopes(in.Manifest, a.Scopes)

	stored := storedPayload{
		Ref:                in.Ref,
		Client:             a.Client,
		TokenEndpoint:      a.TokenEndpoint,
		RevocationEndpoint: a.Revocation,
		Issuer:             a.Issuer,
		Resource:           a.Resource,
		AccessToken:        token.AccessToken,
		TokenType:          token.TokenType,
		RefreshToken:       token.RefreshToken,
		ExpiresAt:          token.expiresAt(in.Manifest, now),
		Scopes:             account.Scopes,
	}
	if stored.RefreshToken != "" {
		stored.RefreshExpiresAt = refreshExpiresAt(in.Manifest, now)
	}
	payload, err := json.Marshal(stored)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: payload}, account, nil
}

// spend records state as used until expires and reports whether it was unused. Entries
// past their expiry are dropped on the way, so the map holds at most attemptTTL of
// attempts.
func (s *Scheme) spend(state string, expires, now time.Time) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	maps.DeleteFunc(s.spent, func(_ string, until time.Time) bool { return !now.Before(until) })
	if _, used := s.spent[state]; used {
		return false
	}
	s.spent[state] = expires
	return true
}

// separator joins and splits scopes: the manifest's, or a space (RFC 6749 section 3.3).
func separator(m core.ResolvedManifest) string {
	if m.Scopes.Separator == "" {
		return " "
	}
	return m.Scopes.Separator
}

// random is randomBytes from crypto/rand, base64url-encoded without padding, which is the
// verifier's alphabet (RFC 7636 section 4.1) and safe in a query.
func random() (string, error) {
	b := make([]byte, randomBytes)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}
