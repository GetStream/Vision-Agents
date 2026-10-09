// Package oauth2cc is the oauth2_client_credentials scheme: the OAuth 2.0 client credentials
// grant (RFC 6749 section 4.4), for every provider a manifest describes.
//
// Begin is Done, since nobody consents. Complete takes the client id and secret the
// developer supplies, asks the manifest's token endpoint for a first access token, so a
// wrong secret is refused at once, and seals the client and the token. Retrieve hands that
// token out until it is due, then asks for a new one with the same client: there is no
// refresh token to spend. Wrap and Classify are oauth2code's. Revoke revokes the access
// token (RFC 7009); the client itself lives on at the provider.
package oauth2cc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Name is the registry name manifests list under schemes: RFC 6749 section 4.4.2's
// grant_type value, client_credentials, under the oauth2_ prefix oauth2_code has. The
// architecture doc's scheme list and its Salesforce manifest spell it this way
// («Adapters», «A manifest, for scale» on connectors/planning).
const Name = "oauth2_client_credentials"

// The keys of core.CompleteInput.Supplied this scheme reads: RFC 6749 section 2.3.1's
// parameter names for the client's credentials.
const (
	SuppliedClientID     = "client_id"
	SuppliedClientSecret = "client_secret"
)

const (
	// payloadVersion is the shape of payload. A new shape is a new version, so a sealed
	// blob is always read as what it was written as.
	payloadVersion = 1
	// defaultMargin is how long before expiry a token is replaced when the manifest sets no
	// refresh.margin: oauth2code's default, the prototype's rule (internal/connectors/
	// runtime.go:78 on codex/connector-support at cf62af0d), so both OAuth schemes renew
	// alike.
	defaultMargin = time.Minute
)

// Errors Revoke returns that no retry changes, so neither is a *core.OutcomeError.
var (
	// ErrNoRevocationEndpoint is Revoke for a manifest with no endpoints.revoke: nothing
	// was sent, and the access token works until it expires.
	ErrNoRevocationEndpoint = errors.New("oauth2cc: the manifest names no revocation endpoint")
	// ErrTokenTypeNotRevocable is the revocation endpoint answering unsupported_token_type
	// (RFC 7009 section 2.2.1): it does not revoke access tokens, and nothing was revoked.
	ErrTokenTypeNotRevocable = errors.New("oauth2cc: the provider does not revoke access tokens")
)

// ErrNoLifetime is a token response without expires_in for a manifest without
// refresh.access_ttl. Such a token would be handed out until the provider ends it, and
// nothing renews it then, though a new one costs one request. So the scheme refuses it: at
// Complete nothing is connected, and the manifest needs an access_ttl.
var ErrNoLifetime = errors.New("oauth2cc: the token response has no expires_in and the manifest sets no refresh.access_ttl, so the token could never be renewed in time")

// refusedClient are the token endpoint error codes that say the client itself is refused,
// so only new client credentials, a reconnect, help: RFC 6749 section 5.2's invalid_client
// («Client authentication failed») and unauthorized_client («not authorized to use this
// authorization grant type»), and invalid_client_id, which is what Salesforce's token
// endpoint answered to an unknown client_id with grant_type=client_credentials (POST
// https://login.salesforce.com/services/oauth2/token at 2026-10-06T19:21:59Z, HTTP 400) and
// what Slack's oauth.v2.access lists (fakeprovider's tokenError cites it). For oauth2_code
// the same codes are a refusal that spends nothing (oauth2code's errorCodes), because the
// client is not the grant there; here it is.
var refusedClient = map[string]bool{"invalid_client": true, "unauthorized_client": true, "invalid_client_id": true}

// Config is what the scheme depends on.
type Config struct {
	// HTTP carries every token and revocation request. In the router it is
	// egress.NewClient(timeout, nil); tests pass the fake provider's Client.
	HTTP *http.Client
	// Now is the clock a token's expiry is judged by; nil is time.Now. Tests move it.
	Now func() time.Time
	// PublicEndpoint checks the token and revocation endpoints before anything is sent, as
	// oauth2code checks every endpoint; nil is egress.ValidatePublicHTTPSURL. Tests that run
	// against a loopback fake pass one that lets the fake's host through.
	PublicEndpoint func(ctx context.Context, raw string) error
}

// Scheme is the oauth2_client_credentials scheme. It holds no state of its own and is safe
// for concurrent use.
type Scheme struct {
	now func() time.Time
	// oauth is the oauth2_code scheme this one sends token requests through and reads
	// answers with, so both OAuth schemes authenticate a client and classify an answer the
	// same way.
	oauth *oauth2code.Scheme
}

var (
	_ core.Scheme        = (*Scheme)(nil)
	_ core.Fingerprinter = (*Scheme)(nil)
)

// New checks cfg and returns the scheme.
func New(cfg Config) (*Scheme, error) {
	if cfg.HTTP == nil {
		return nil, stack.Wrap(errors.New("oauth2cc: Config.HTTP is required"))
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	oauth, err := oauth2code.New(oauth2code.Config{HTTP: cfg.HTTP, Now: cfg.Now, PublicEndpoint: cfg.PublicEndpoint})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Scheme{now: cfg.Now, oauth: oauth}, nil
}

// Name is Name.
func (*Scheme) Name() string {
	return Name
}

// payload is the sealed payload of a connection's StoredCredentials: the client, which is
// the long-lived credential, and the access token last issued to it.
type payload struct {
	ClientID     string    `json:"client_id"`
	ClientSecret string    `json:"client_secret"`
	AccessToken  string    `json:"access_token"`
	ExpiresAt    time.Time `json:"expires_at,omitzero"`
}

// Begin is Done: the client is supplied, nobody consents. It refuses a manifest this scheme
// cannot get a token from, before anything is supplied.
func (*Scheme) Begin(_ context.Context, in core.BeginInput) (core.BeginOutput, error) {
	if _, err := tokenEndpoint(in.Manifest); err != nil {
		return core.BeginOutput{}, err
	}
	if _, err := authMethod(in.Manifest); err != nil {
		return core.BeginOutput{}, err
	}
	return core.BeginOutput{Done: true}, nil
}

// Complete checks the supplied client, gets a first access token with it and seals both.
// A token endpoint that refuses the client fails here, so a connection is never made with
// credentials that do not work. Its errors name what is wrong, never the value.
func (s *Scheme) Complete(ctx context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	for name := range in.Supplied {
		if name != SuppliedClientID && name != SuppliedClientSecret {
			return core.StoredCredentials{}, core.AccountInfo{}, stack.Wrap(fmt.Errorf("oauth2cc: %q is not a value %s takes; it takes %s and %s", name, Name, SuppliedClientID, SuppliedClientSecret))
		}
	}
	client := payload{ClientID: in.Supplied[SuppliedClientID], ClientSecret: in.Supplied[SuppliedClientSecret]}
	if err := client.check(); err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	next, raw, err := s.mint(ctx, in.Manifest, client)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	// No callback: everything the manifest captures comes from the token response.
	account, err := in.Manifest.Apply(nil, raw)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, stack.Wrap(fmt.Errorf("oauth2cc: %w", err))
	}
	account.Scopes = scopes(in.Manifest, raw)
	stored, err := seal(next)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return stored, account, nil
}

// Retrieve hands out the access token in stored until it is inside the margin of its expiry
// or expires at or before opts.ValidUntil, and then asks the token endpoint for a new one
// with the stored client (RFC 6749 section 4.4.2). Every token mint issues has an expiry
// (ErrNoLifetime); one stored without it is due at once. A failed request returns a
// *core.OutcomeError and no StoredCredentials, with the old token beside it while that has
// not expired.
//
// A token the provider refused (opts.Refused) is replaced at once, and never handed back
// when the new request fails.
//
// opts.Checkpoint is never called: a client credentials request spends nothing. No refresh
// token goes out (section 4.4.3: none is issued), the client's secret stays valid whatever
// the answer, and a second request only mints another access token. So a lost answer is
// safe to follow with the same request, and committing needs_reauthorization before it,
// which is what the checkpoint does (internal/connectors/resolver), would turn a dropped
// connection into a reconnect nobody needed.
func (s *Scheme) Retrieve(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest, opts core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	current, err := open(stored)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	now := s.now()
	// The token must outlive both the margin and the call (opts.ValidUntil).
	due := now.Add(margin(m))
	if opts.ValidUntil.After(due) {
		due = opts.ValidUntil
	}
	// One the provider refused is replaced whatever its expiry says (opts.Refused): minting
	// another costs nothing, so a refusal never needs a reconnect.
	if !opts.Refused && !current.ExpiresAt.IsZero() && due.Before(current.ExpiresAt) {
		return credential(current), stored, nil
	}
	next, _, err := s.mint(ctx, m, current)
	if err != nil {
		if !opts.Refused && s.now().Before(current.ExpiresAt) {
			return credential(current), core.StoredCredentials{}, err
		}
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	renewed, err := seal(next)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return credential(next), renewed, nil
}

// Wrap puts the access token on every request as a bearer token (RFC 6750 section 2.1),
// with oauth2code's Bearer. A credential this scheme did not issue fails each request
// instead.
func (*Scheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var secret struct {
		AccessToken string `json:"access_token"`
	}
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &secret) != nil {
		secret.AccessToken = ""
	}
	return oauth2code.Bearer(base, secret.AccessToken)
}

// Classify is oauth2code's: a resource answers a token from this grant as it answers any
// bearer token (RFC 6750 section 3.1).
func (s *Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	return s.oauth.Classify(resp, body, err)
}

// Revoke asks the manifest's revoke endpoint to revoke the access token (RFC 7009 section
// 2.1, token_type_hint access_token), authenticated as the stored client. There is no
// refresh token, and the client is not the router's to revoke: it lives on at the provider,
// so whoever holds its secret can still get a token until the customer rotates it there.
//
// nil means the endpoint answered 200, which section 2.2 also answers for a token it did
// not know. ErrNoRevocationEndpoint and ErrTokenTypeNotRevocable say nothing was revoked;
// any other refusal is a *core.OutcomeError.
func (s *Scheme) Revoke(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest) error {
	current, err := open(stored)
	if err != nil {
		return err
	}
	endpoint := m.Endpoints["revoke"]
	if endpoint == "" {
		return stack.Wrap(ErrNoRevocationEndpoint)
	}
	method, err := authMethod(m)
	if err != nil {
		return err
	}
	form := url.Values{}
	form.Set("token", current.AccessToken)
	form.Set("token_type_hint", "access_token")
	response, raw, err := s.oauth.TokenRequest(ctx, endpoint, form, oauth2code.Client{ID: current.ClientID, Secret: current.ClientSecret, AuthMethod: method})
	if err == nil && response.StatusCode == http.StatusOK {
		return nil
	}
	if err == nil && errorCode(raw) == "unsupported_token_type" {
		return stack.Wrap(fmt.Errorf("%w (HTTP %d)", ErrTokenTypeNotRevocable, response.StatusCode))
	}
	outcome := s.Classify(response, raw, err)
	if outcome.Kind == core.OutcomeOK {
		// A refusal Classify has no more to say about, such as a 404: nothing was revoked.
		outcome.Kind = core.OutcomeTransient
	}
	if err == nil {
		err = &oauth2code.TokenError{Status: response.StatusCode, Code: errorCode(raw)}
	}
	return &core.OutcomeError{Outcome: outcome, Err: stack.Wrap(fmt.Errorf("oauth2cc: revoke: %w", err))}
}

// mint is one client credentials request (RFC 6749 section 4.4.2) and the payload it leaves:
// client with the new access token. raw is the token response, for the manifest's capture
// rules. A failure is a *core.OutcomeError.
func (s *Scheme) mint(ctx context.Context, m core.ResolvedManifest, client payload) (payload, json.RawMessage, error) {
	endpoint, err := tokenEndpoint(m)
	if err != nil {
		return payload{}, nil, err
	}
	method, err := authMethod(m)
	if err != nil {
		return payload{}, nil, err
	}
	form := url.Values{}
	form.Set("grant_type", "client_credentials")
	// No scope: section 4.4.2 makes it OPTIONAL, and the manifest's scopes.list is what a
	// person consents to under oauth2_code (Salesforce's lists refresh_token, which this
	// grant never issues). The client's own registration at the provider decides what its
	// tokens may do, as a Salesforce connected app's selected scopes do («Configuring the
	// flow», developer.salesforce.com/blogs/2023/03/using-the-client-credentials-flow-for-
	// easier-api-authentication, read 2026-10-06).
	if resource := m.Endpoints["resource"]; resource != "" {
		// RFC 8707 section 2.2: the resource of the token asked for, on any token request,
		// as oauth2code sends it with a code and a refresh.
		form.Set("resource", resource)
	}
	response, raw, err := s.oauth.TokenRequest(ctx, endpoint, form, oauth2code.Client{ID: client.ClientID, Secret: client.ClientSecret, AuthMethod: method})
	if err == nil && response.StatusCode == http.StatusOK && errorCode(raw) == "" {
		var token struct {
			AccessToken string      `json:"access_token"`
			ExpiresIn   json.Number `json:"expires_in"`
		}
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		if decoder.Decode(&token) == nil && token.AccessToken != "" {
			next := client
			next.AccessToken = token.AccessToken
			next.ExpiresAt = expiresAt(m, token.ExpiresIn, s.now())
			if next.ExpiresAt.IsZero() {
				return payload{}, nil, stack.Wrap(ErrNoLifetime)
			}
			return next, raw, nil
		}
		// RFC 6749 section 5.1: access_token is REQUIRED. Nothing was spent, so asking again
		// may do better.
		return payload{}, nil, &core.OutcomeError{
			Outcome: core.Outcome{Kind: core.OutcomeTransient},
			Err:     stack.Wrap(errors.New("oauth2cc: token: HTTP 200 without a readable access_token")),
		}
	}
	outcome := s.refusal(response, raw, err)
	if err == nil {
		err = &oauth2code.TokenError{Status: response.StatusCode, Code: errorCode(raw)}
	}
	return payload{}, nil, &core.OutcomeError{Outcome: outcome, Err: stack.Wrap(fmt.Errorf("oauth2cc: token: %w", err))}
}

// refusal is the outcome of a token request that did not return a token. Classify reads it
// first; then a refused client is InvalidGrant (refusedClient, and a 401, which section 5.2
// lets a server answer invalid_client with), since the client is the whole grant; and what
// Classify calls Uncertain, or OK, is Transient, since a client credentials request spends
// nothing that a retry could replay (Retrieve).
func (s *Scheme) refusal(response *http.Response, raw []byte, err error) core.Outcome {
	outcome := s.Classify(response, raw, err)
	switch {
	case outcome.Kind == core.OutcomeRateLimited:
		return outcome
	case response != nil && (refusedClient[errorCode(raw)] || response.StatusCode == http.StatusUnauthorized):
		return core.Outcome{Kind: core.OutcomeInvalidGrant}
	case outcome.Kind == core.OutcomeInvalidGrant || outcome.Kind == core.OutcomeScopeRequired:
		return outcome
	}
	return core.Outcome{Kind: core.OutcomeTransient, RetryAfter: outcome.RetryAfter}
}

// check refuses a client RFC 6749 cannot send: an id and a secret are each *VSCHAR (Appendix
// A.1, A.2: %x20-7E), and this grant needs both, since section 4.4 is for confidential
// clients only. Its errors name what is wrong, never the value.
func (p payload) check() error {
	switch {
	case p.ClientID == "":
		return stack.Wrap(errors.New("oauth2cc: client_id is required"))
	case !vschar(p.ClientID):
		return stack.Wrap(errors.New("oauth2cc: client_id has a character outside RFC 6749's VSCHAR (%x20-7E)"))
	case p.ClientSecret == "":
		return stack.Wrap(errors.New("oauth2cc: client_secret is required: the client credentials grant is for confidential clients only"))
	case !vschar(p.ClientSecret):
		return stack.Wrap(errors.New("oauth2cc: client_secret has a character outside RFC 6749's VSCHAR (%x20-7E)"))
	}
	return nil
}

func vschar(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] < 0x20 || value[i] > 0x7e {
			return false
		}
	}
	return true
}

// tokenEndpoint is the manifest's endpoints.token. Nothing is discovered: a client
// credentials manifest pins where its tokens come from.
func tokenEndpoint(m core.ResolvedManifest) (string, error) {
	endpoint := m.Endpoints["token"]
	if endpoint == "" {
		return "", stack.Wrap(fmt.Errorf("oauth2cc: manifest %q has no endpoints.token", m.ConnectorID))
	}
	return endpoint, nil
}

// authMethod is how the client authenticates at the token endpoint: the manifest's
// client.auth_method, else client_secret_basic, the method RFC 6749 section 2.3.1 says every
// server «MUST support» (and calls the body form «NOT RECOMMENDED»). Only the two secret
// methods are taken: section 4.4.2 «MUST authenticate», so none is out, and
// private_key_jwt and tls_client_auth have no key here (oauth2code's supportedMethods).
func authMethod(m core.ResolvedManifest) (core.ClientAuthMethod, error) {
	switch m.Client.AuthMethod {
	case "":
		return core.AuthClientSecretBasic, nil
	case core.AuthClientSecretBasic, core.AuthClientSecretPost:
		return m.Client.AuthMethod, nil
	}
	return "", stack.Wrap(fmt.Errorf("oauth2cc: client.auth_method %q is not one this scheme sends; it takes %s and %s",
		m.Client.AuthMethod, core.AuthClientSecretBasic, core.AuthClientSecretPost))
}

// expiresAt is when a token issued at now expires: expires_in (RFC 6749 section 5.1,
// RECOMMENDED there), else the manifest's refresh.access_ttl, else zero, which mint refuses
// (ErrNoLifetime).
func expiresAt(m core.ResolvedManifest, expiresIn json.Number, now time.Time) time.Time {
	if seconds, err := expiresIn.Int64(); err == nil && seconds > 0 {
		return now.Add(time.Duration(seconds) * time.Second)
	}
	if m.Refresh.AccessTTL > 0 {
		return now.Add(time.Duration(m.Refresh.AccessTTL))
	}
	return time.Time{}
}

// scopes is what the token response says was granted (RFC 6749 section 5.1, scope), split
// on the manifest's scopes.separator, else a space (section 3.3). Without one, nothing is
// known: no scope was asked for (mint).
func scopes(m core.ResolvedManifest, raw []byte) []string {
	var token struct {
		Scope string `json:"scope"`
	}
	if json.Unmarshal(raw, &token) != nil || token.Scope == "" {
		return nil
	}
	separator := m.Scopes.Separator
	if separator == "" {
		separator = " "
	}
	var granted []string
	for _, scope := range strings.Split(token.Scope, separator) {
		if scope = strings.TrimSpace(scope); scope != "" {
			granted = append(granted, scope)
		}
	}
	return granted
}

// margin is the manifest's refresh.margin, else defaultMargin.
func margin(m core.ResolvedManifest) time.Duration {
	if m.Refresh.Margin > 0 {
		return time.Duration(m.Refresh.Margin)
	}
	return defaultMargin
}

// errorCode is the error member of a JSON object body (RFC 6749 section 5.2), or "".
func errorCode(raw []byte) string {
	var body struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(raw, &body) != nil {
		return ""
	}
	return body.Error
}

// Fingerprints names the access token last issued in stored, by core.Fingerprint, with its
// expiry, so a renewal shows in the audit and the log as oauth2_code's does (AI-990). The
// client credentials grant issues no refresh token (RFC 6749 section 4.4.3), and the client
// secret is the connection's credential, not a token, so neither is named.
func (*Scheme) Fingerprints(stored core.StoredCredentials) (core.CredentialFingerprints, error) {
	p, err := open(stored)
	if err != nil {
		return core.CredentialFingerprints{}, err
	}
	return core.CredentialFingerprints{Access: core.Fingerprint(p.AccessToken), AccessExpiresAt: p.ExpiresAt}, nil
}

// open reads the payload this scheme sealed. Its errors never quote the payload.
func open(stored core.StoredCredentials) (payload, error) {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return payload{}, stack.Wrap(fmt.Errorf("oauth2cc: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion))
	}
	var p payload
	if json.Unmarshal(stored.Payload, &p) != nil || p.AccessToken == "" {
		return payload{}, stack.Wrap(errors.New("oauth2cc: stored credentials payload is unreadable or has no access token"))
	}
	if err := p.check(); err != nil {
		return payload{}, err
	}
	return p, nil
}

func seal(p payload) (core.StoredCredentials, error) {
	raw, err := json.Marshal(p)
	if err != nil {
		return core.StoredCredentials{}, stack.Wrap(err)
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: raw}, nil
}

// credential is the access token in p as a core.AccessCredential, read back by Wrap. The
// client stays in the stored credentials: a request needs only the token.
func credential(p payload) core.AccessCredential {
	secret, _ := json.Marshal(map[string]string{"access_token": p.AccessToken})
	return core.NewAccessCredential(Name, p.ExpiresAt, secret)
}
