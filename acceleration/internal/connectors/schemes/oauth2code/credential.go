package oauth2code

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// defaultMargin is how long before expiry an access token is renewed when the manifest
// sets no refresh.margin. It is the prototype's rule, refresh when under a minute is left
// (internal/connectors/runtime.go:78 on codex/connector-support at cf62af0d), which also
// leaves room for clock skew and one slow token request before the old token dies.
const defaultMargin = time.Minute

// ErrNoRevocationEndpoint is Revoke when neither the manifest nor the authorization
// server's metadata names a revocation endpoint (RFC 8414 section 2, revocation_endpoint):
// nothing was sent, and the grant lives on at the provider.
var ErrNoRevocationEndpoint = errors.New("oauth2code: the provider has no revocation endpoint")

// ErrTokenTypeNotRevocable is Revoke when the revocation endpoint answers
// unsupported_token_type (RFC 7009 section 2.2.1): it does not revoke that kind of token, so
// asking again will not help, and nothing was revoked.
var ErrTokenTypeNotRevocable = errors.New("oauth2code: the provider does not revoke this kind of token")

// errNoAccessToken is what a request carried by Wrap fails with when the credential holds
// no access token, rather than leave without one.
var errNoAccessToken = errors.New("oauth2code: the credential has no access token")

// errUnknownClient is Export for a credential that does not say which client its grant was
// issued to, so whose app it is cannot be told.
var errUnknownClient = errors.New("oauth2code: the credential does not say which client its grant was issued to")

// Retrieve returns the access token in stored, renewed first when it is inside the
// margin of its expiry, expires at or before opts.ValidUntil, or the provider refused it
// (opts.Refused) (RFC 6749 section 6). The
// StoredCredentials that come back are stored itself when nothing was renewed, and new ones when a refresh succeeded; a failed refresh
// returns no StoredCredentials, stored is never written to, and the error is a *core.OutcomeError the resolver acts on. When the
// refresh failed inside the margin, before the access token expired, that still valid token
// comes back with the error, so a provider's bad minute is not a failed call: the resolver
// can use it and still act on the outcome. Once the token has expired, or the provider refused
// it (opts.Refused), there is none.
//
// When a refresh's answer is Uncertain (lost, or a 5xx that may have rotated the token) and
// the manifest gives the provider a refresh.grace, the same refresh token is sent once more
// while the window opened by the first attempt is still running: a retired token keeps
// working inside it, so the retry cannot be the replay RFC 9700 section 4.14.2 revokes a
// grant for. Without a grace, a retry could be exactly that, so there is none.
func (s *Scheme) Retrieve(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest, opts core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	current, err := open(stored)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	now := s.cfg.Now()
	// The token must outlive both the margin and the call (opts.ValidUntil).
	due := now.Add(margin(m))
	if opts.ValidUntil.After(due) {
		due = opts.ValidUntil
	}
	// A token with no known expiry is never renewed early: there is no margin to be inside.
	// One the provider refused is renewed whatever its expiry says (opts.Refused).
	if !opts.Refused && (current.ExpiresAt.IsZero() || due.Before(current.ExpiresAt)) {
		return credential(current), stored, nil
	}
	if current.RefreshToken == "" {
		if current.ExpiresAt.IsZero() || now.Before(current.ExpiresAt) {
			// Still valid, and nothing to renew it with.
			return credential(current), stored, nil
		}
		return core.AccessCredential{}, core.StoredCredentials{}, &core.OutcomeError{
			Outcome: core.Outcome{Kind: core.OutcomeInvalidGrant},
			Err:     errors.New("oauth2code: the access token expired and the grant has no refresh token"),
		}
	}
	next, err := s.refresh(ctx, m, current, opts.Checkpoint)
	if err != nil {
		// A token the provider refused is not handed back, however long it has left.
		if !opts.Refused && s.cfg.Now().Before(current.ExpiresAt) {
			return credential(current), core.StoredCredentials{}, err
		}
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	payload, err := json.Marshal(next)
	if err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return credential(next), core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: payload}, nil
}

// Wrap puts the access token on every request as a bearer token (RFC 6750 section 2.1).
// Every token this scheme hands out is sent that way, whatever token_type the provider named
// it (a provider-shaped value such as "bot" is still a bearer token on the wire). A
// credential without an access token fails each request instead.
func (s *Scheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var secret accessSecret
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &secret) != nil {
		return refuse{}
	}
	return Bearer(base, secret.AccessToken)
}

// Export is the access token as a bearer token (RFC 6750 section 2.1), as Wrap sends it, and
// the registration of the client it was issued to, for the caller to decide whose app that
// is. A credential this scheme did not issue, one without an access token, or one whose
// client it cannot tell, is refused.
func (s *Scheme) Export(c core.AccessCredential) (core.ExportedCredential, error) {
	var secret accessSecret
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &secret) != nil || secret.AccessToken == "" {
		return core.ExportedCredential{}, errNoAccessToken
	}
	if secret.Client == "" {
		return core.ExportedCredential{}, errUnknownClient
	}
	return core.ExportedCredential{Header: "Authorization", Value: "Bearer " + secret.AccessToken,
		ExpiresAt: c.ExpiresAt, Client: secret.Client}, nil
}

// Bearer is the RoundTripper that puts accessToken on a clone of every request as a bearer
// token (RFC 6750 section 2.1), for this scheme's Wrap and another OAuth scheme's (oauth2cc).
// An empty accessToken fails every request instead of sending it without one.
func Bearer(base http.RoundTripper, accessToken string) http.RoundTripper {
	if accessToken == "" {
		return refuse{}
	}
	return bearer{base: base, authorization: "Bearer " + accessToken}
}

// Revoke asks the provider to revoke the grant (RFC 7009) at the manifest's revoke
// endpoint, else the one the authorization server's metadata named. It sends the refresh
// token when there is one, since section 2 makes revoking a refresh token a MUST and an
// access token only a SHOULD, and the access token otherwise.
//
// A nil error means the endpoint answered 200, which section 2.2 also answers for a token
// it did not know: it is not proof that anything was revoked. Nor does it end the access
// token: section 2.1 has the server «also invalidate all access tokens based on the same
// authorization grant» only if it «supports the revocation of access tokens», so the access
// token may work until it expires. ErrTokenTypeNotRevocable is the server saying it does not
// revoke that kind of token (section 2.2.1, unsupported_token_type). Any other refusal is a
// *core.OutcomeError; section 2.2.1 answers 503 when the client should try again.
func (s *Scheme) Revoke(ctx context.Context, stored core.StoredCredentials, m core.ResolvedManifest) error {
	current, err := open(stored)
	if err != nil {
		return err
	}
	endpoint := firstSet(m.Endpoints["revoke"], current.RevocationEndpoint)
	if endpoint == "" {
		return ErrNoRevocationEndpoint
	}
	// Held to the egress policy as discovery holds every endpoint (checkEndpoint).
	if err := s.checkEndpoint(ctx, endpoint); err != nil {
		return err
	}
	c, err := s.clientSecret(ctx, current.Ref, m, current.Client)
	if err != nil {
		return err
	}
	form := url.Values{}
	// RFC 7009 section 2.1: token, and token_type_hint naming which kind it is.
	if current.RefreshToken != "" {
		form.Set("token", current.RefreshToken)
		form.Set("token_type_hint", "refresh_token")
	} else {
		form.Set("token", current.AccessToken)
		form.Set("token_type_hint", "access_token")
	}
	response, raw, err := s.tokenPost(ctx, endpoint, form, c)
	if err == nil && response.StatusCode == http.StatusOK {
		return nil
	}
	if err == nil && errorCode(raw) == "unsupported_token_type" {
		return fmt.Errorf("%w (%s, HTTP %d)", ErrTokenTypeNotRevocable, form.Get("token_type_hint"), response.StatusCode)
	}
	outcome := s.Classify(response, raw, err)
	if outcome.Kind == core.OutcomeOK {
		// A refusal Classify has no more to say about, such as a 404: nothing was revoked,
		// and nothing else changed.
		outcome.Kind = core.OutcomeTransient
	}
	if err == nil {
		err = &TokenError{Status: response.StatusCode, Code: errorCode(raw)}
	}
	return &core.OutcomeError{Outcome: outcome, Err: fmt.Errorf("oauth2code: revoke: %w", err)}
}

// refresh is the refresh_token grant (RFC 6749 section 6) with one grace retry, and the
// payload it leaves.
func (s *Scheme) refresh(ctx context.Context, m core.ResolvedManifest, current storedPayload, checkpoint func() error) (storedPayload, error) {
	// The manifest's refresh endpoint, when a provider renews somewhere else, then the token
	// endpoint the code was redeemed at.
	endpoint := firstSet(m.Endpoints["refresh"], current.TokenEndpoint)
	// Held to the egress policy as discovery holds every endpoint (checkEndpoint).
	if err := s.checkEndpoint(ctx, endpoint); err != nil {
		return storedPayload{}, err
	}
	// A preregistered client's secret is looked up again, as at Complete, so a rotated one
	// is used at once.
	c, err := s.clientSecret(ctx, current.Ref, m, current.Client)
	if err != nil {
		return storedPayload{}, err
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", current.RefreshToken)
	if m.Scopes.SendOnRefresh && len(current.Scopes) > 0 {
		// RFC 6749 section 6: scope is OPTIONAL and «MUST NOT include any scope not
		// originally granted», so what is sent is the granted scopes, when the manifest says
		// the provider needs them.
		form.Set("scope", strings.Join(current.Scopes, separator(m)))
	}
	if current.Resource != "" {
		// RFC 8707 section 2.2: the resource of the access token asked for, on a refresh too.
		form.Set("resource", current.Resource)
	}
	// The resolver's checkpoint commits that the outcome is not known yet before the refresh
	// token leaves, so an answer that never arrives is never followed by the same token. The
	// grace retry below needs none of its own: the first attempt already committed it.
	if checkpoint != nil {
		if err := checkpoint(); err != nil {
			return storedPayload{}, err
		}
	}
	start := s.cfg.Now()
	token, err := s.redeem(ctx, endpoint, form, c)
	var failed *core.OutcomeError
	if errors.As(err, &failed) && failed.Outcome.Kind == core.OutcomeUncertain &&
		m.Refresh.Grace > 0 && s.cfg.Now().Sub(start) < time.Duration(m.Refresh.Grace) {
		token, err = s.redeem(ctx, endpoint, form, c)
	}
	if err != nil {
		return storedPayload{}, err
	}

	now := s.cfg.Now()
	next := current
	next.AccessToken = token.AccessToken
	next.TokenType = token.TokenType
	next.ExpiresAt = token.expiresAt(m, now)
	next.Scopes = token.scopes(m, current.Scopes)
	if token.RefreshToken != "" {
		// RFC 6749 section 6: with a new refresh token «the client MUST discard the old
		// refresh token and replace it with the new one». Without one the old one stays.
		next.RefreshToken = token.RefreshToken
		next.RefreshExpiresAt = refreshExpiresAt(m, now)
	}
	s.warnIfLastRefresh(m, next)
	return next, nil
}

// redeem sends one refresh request and reads its answer: a token, or a *core.OutcomeError.
func (s *Scheme) redeem(ctx context.Context, endpoint string, form url.Values, c client) (tokenResponse, error) {
	response, raw, err := s.tokenPost(ctx, endpoint, cloneValues(form), c)
	outcome := s.Classify(response, raw, err)
	if outcome.Kind == core.OutcomeOK && errorCode(raw) != "" {
		// An error member Classify does not name, at any status: the token endpoint refused
		// the refresh (RFC 6749 section 5.2), so nothing was spent. Some servers send it with
		// 200, as the fake's Slack shape does.
		outcome.Kind = core.OutcomeTransient
	}
	if outcome.Kind == core.OutcomeOK {
		if response.StatusCode >= 200 && response.StatusCode < 300 {
			body, err := decodeObject(raw)
			if err == nil {
				if token, err := parseToken(body); err == nil {
					return token, nil
				}
			}
			// A success without a token it can read: the server may have rotated the
			// refresh token all the same, as the prototype assumed (internal/mcp/oauth.go:
			// 586-588 at cf62af0d).
			return tokenResponse{}, &core.OutcomeError{
				Outcome: core.Outcome{Kind: core.OutcomeUncertain},
				Err:     fmt.Errorf("oauth2code: refresh: HTTP %d without a readable token", response.StatusCode),
			}
		}
		// A refusal Classify has no more to say about, such as a bare 400: the server said
		// no, so nothing was spent (RFC 6749 section 5.1 answers success with 200).
		outcome.Kind = core.OutcomeTransient
	}
	if err == nil {
		err = &TokenError{Status: response.StatusCode, Code: errorCode(raw)}
	}
	return tokenResponse{}, &core.OutcomeError{Outcome: outcome, Err: fmt.Errorf("oauth2code: refresh: %w", err)}
}

// warnIfLastRefresh logs when the refresh token, by the manifest's refresh.refresh_ttl, dies
// before the access token just issued is due for renewal: the next renewal will fail and
// only a reconnect helps. It names the connector and the connection, never a token.
func (s *Scheme) warnIfLastRefresh(m core.ResolvedManifest, payload storedPayload) {
	if payload.RefreshExpiresAt.IsZero() || payload.ExpiresAt.IsZero() || !payload.RefreshExpiresAt.Before(payload.ExpiresAt.Add(-margin(m))) {
		return
	}
	s.cfg.Logger.Warn("oauth2code: the refresh token expires before the next refresh; the connection will need a reconnect",
		"connector", m.ConnectorID, "connection", payload.Ref.ConnectionID, "refresh_expires_at", payload.RefreshExpiresAt)
}

// open reads the payload of the StoredCredentials this scheme sealed. Its errors never quote the payload.
func open(stored core.StoredCredentials) (storedPayload, error) {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return storedPayload{}, fmt.Errorf("oauth2code: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion)
	}
	var out storedPayload
	if json.Unmarshal(stored.Payload, &out) != nil || out.AccessToken == "" {
		return storedPayload{}, errors.New("oauth2code: stored credentials payload is unreadable or has no access token")
	}
	return out, nil
}

// credential is the access token in payload as a core.AccessCredential, read back by Wrap,
// with the registration of the client it was issued to, read back by Export. Never the
// refresh token.
func credential(payload storedPayload) core.AccessCredential {
	secret, _ := json.Marshal(accessSecret{AccessToken: payload.AccessToken, Client: payload.Client.RegistrationMethod})
	return core.NewAccessCredential(Name, payload.ExpiresAt, secret)
}

// accessSecret is an AccessCredential's secret.
type accessSecret struct {
	AccessToken string                        `json:"access_token"`
	Client      core.ClientRegistrationMethod `json:"client,omitempty"`
}

// margin is the manifest's refresh.margin, else defaultMargin.
func margin(m core.ResolvedManifest) time.Duration {
	if m.Refresh.Margin > 0 {
		return time.Duration(m.Refresh.Margin)
	}
	return defaultMargin
}

// refreshExpiresAt is when a refresh token issued at now dies by the manifest's
// refresh.refresh_ttl, counted from each issue (a rotated token starts its own), or zero
// when the manifest does not say.
func refreshExpiresAt(m core.ResolvedManifest, now time.Time) time.Time {
	if m.Refresh.RefreshTTL <= 0 {
		return time.Time{}
	}
	return now.Add(time.Duration(m.Refresh.RefreshTTL))
}

// cloneValues copies a form, since tokenPost adds the client's parameters to the one it is
// given and a retry must send what the first attempt did.
func cloneValues(form url.Values) url.Values {
	out := make(url.Values, len(form))
	for k, v := range form {
		out[k] = append([]string(nil), v...)
	}
	return out
}

// bearer is the RoundTripper Wrap returns.
type bearer struct {
	base          http.RoundTripper
	authorization string
}

// RoundTrip sets the header on a clone: net/http's RoundTripper contract is that
// «RoundTrip should not modify the request».
func (b bearer) RoundTrip(r *http.Request) (*http.Response, error) {
	out := r.Clone(r.Context())
	out.Header.Set("Authorization", b.authorization)
	return b.base.RoundTrip(out)
}

type refuse struct{}

func (refuse) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		// net/http's RoundTripper contract: it «must always close the body, including on
		// errors».
		_ = r.Body.Close()
	}
	return nil, errNoAccessToken
}
