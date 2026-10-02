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

// errNoAccessToken is what a request carried by Wrap fails with when the credential holds
// no access token, rather than leave without one.
var errNoAccessToken = errors.New("oauth2code: the credential has no access token")

// Mint returns the access token in m, renewed first when it is inside the margin of its
// expiry (RFC 6749 section 6). The Material that comes back is m itself when nothing was
// renewed, and new material when a refresh succeeded; a failed refresh returns no Material,
// m is never written to, and the error is a *core.OutcomeError the resolver acts on.
//
// When a refresh's answer is Uncertain (lost, or a 5xx that may have rotated the token) and
// the manifest gives the provider a refresh.grace, the same refresh token is sent once more
// while the window opened by the first attempt is still running: a retired token keeps
// working inside it, so the retry cannot be the replay RFC 9700 section 4.14.2 revokes a
// grant for. Without a grace, a retry could be exactly that, so there is none.
func (s *Scheme) Mint(ctx context.Context, m core.Material, p core.Profile) (core.Credential, core.Material, error) {
	current, err := open(m)
	if err != nil {
		return core.Credential{}, core.Material{}, err
	}
	now := s.cfg.Now()
	// A token with no known expiry is never renewed early: there is no margin to be inside.
	if current.ExpiresAt.IsZero() || now.Add(margin(p)).Before(current.ExpiresAt) {
		return credential(current), m, nil
	}
	if current.RefreshToken == "" {
		if now.Before(current.ExpiresAt) {
			// Still valid, and nothing to renew it with.
			return credential(current), m, nil
		}
		return core.Credential{}, core.Material{}, &core.OutcomeError{
			Outcome: core.Outcome{Kind: core.OutcomeInvalidGrant},
			Err:     errors.New("oauth2code: the access token expired and the grant has no refresh token"),
		}
	}
	next, err := s.refresh(ctx, p, current)
	if err != nil {
		return core.Credential{}, core.Material{}, err
	}
	payload, err := json.Marshal(next)
	if err != nil {
		return core.Credential{}, core.Material{}, err
	}
	return credential(next), core.Material{Scheme: Name, Version: materialVersion, Payload: payload}, nil
}

// Wrap puts the access token on every request as a bearer token (RFC 6750 section 2.1).
// Every token this scheme mints is sent that way, whatever token_type the provider named
// it (a provider-shaped value such as "bot" is still a bearer token on the wire). A
// credential without an access token fails each request instead.
func (s *Scheme) Wrap(base http.RoundTripper, c core.Credential) http.RoundTripper {
	var secret struct {
		AccessToken string `json:"access_token"`
	}
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &secret) != nil || secret.AccessToken == "" {
		return refuse{}
	}
	return bearer{base: base, authorization: "Bearer " + secret.AccessToken}
}

// Revoke asks the provider to revoke the grant (RFC 7009) at the manifest's revoke
// endpoint, else the one the authorization server's metadata named. It sends the refresh
// token when there is one, since RFC 7009 section 2.1 has the server then «also invalidate
// all access tokens based on the same authorization grant», and the access token otherwise.
// A nil error means the endpoint answered 200, which section 2.2 also answers for a token
// it did not know: it is not proof that anything was revoked. A refusal is a
// *core.OutcomeError; section 2.2.1 answers 503 when the client should try again.
func (s *Scheme) Revoke(ctx context.Context, m core.Material, p core.Profile) error {
	current, err := open(m)
	if err != nil {
		return err
	}
	endpoint := firstSet(p.Endpoints["revoke"], current.RevocationEndpoint)
	if endpoint == "" {
		return ErrNoRevocationEndpoint
	}
	// Held to the egress policy as discovery holds every endpoint (checkEndpoint).
	if err := s.checkEndpoint(ctx, endpoint); err != nil {
		return err
	}
	c, err := s.clientSecret(ctx, current.Ref, p, current.Client)
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
// material it leaves.
func (s *Scheme) refresh(ctx context.Context, p core.Profile, current material) (material, error) {
	// The manifest's refresh endpoint, when a provider renews somewhere else, then the token
	// endpoint the code was redeemed at.
	endpoint := firstSet(p.Endpoints["refresh"], current.TokenEndpoint)
	// Held to the egress policy as discovery holds every endpoint (checkEndpoint).
	if err := s.checkEndpoint(ctx, endpoint); err != nil {
		return material{}, err
	}
	// A preregistered client's secret is looked up again, as at Complete, so a rotated one
	// is used at once.
	c, err := s.clientSecret(ctx, current.Ref, p, current.Client)
	if err != nil {
		return material{}, err
	}
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("refresh_token", current.RefreshToken)
	if p.Scopes.SendOnRefresh && len(current.Scopes) > 0 {
		// RFC 6749 section 6: scope is OPTIONAL and «MUST NOT include any scope not
		// originally granted», so what is sent is the granted scopes, when the manifest says
		// the provider needs them.
		form.Set("scope", strings.Join(current.Scopes, separator(p)))
	}
	if current.Resource != "" {
		// RFC 8707 section 2.2: the resource of the access token asked for, on a refresh too.
		form.Set("resource", current.Resource)
	}
	start := s.cfg.Now()
	token, err := s.redeem(ctx, endpoint, form, c)
	var failed *core.OutcomeError
	if errors.As(err, &failed) && failed.Outcome.Kind == core.OutcomeUncertain &&
		p.Refresh.Grace > 0 && s.cfg.Now().Sub(start) < time.Duration(p.Refresh.Grace) {
		token, err = s.redeem(ctx, endpoint, form, c)
	}
	if err != nil {
		return material{}, err
	}

	now := s.cfg.Now()
	next := current
	next.AccessToken = token.AccessToken
	next.TokenType = token.TokenType
	next.ExpiresAt = token.expiresAt(p, now)
	next.Scopes = token.scopes(p, current.Scopes)
	if token.RefreshToken != "" {
		// RFC 6749 section 6: with a new refresh token «the client MUST discard the old
		// refresh token and replace it with the new one». Without one the old one stays.
		next.RefreshToken = token.RefreshToken
		next.RefreshExpiresAt = refreshExpiresAt(p, now)
	}
	s.warnIfLastRefresh(p, next)
	return next, nil
}

// redeem sends one refresh request and reads its answer: a token, or a *core.OutcomeError.
func (s *Scheme) redeem(ctx context.Context, endpoint string, form url.Values, c client) (tokenResponse, error) {
	response, raw, err := s.tokenPost(ctx, endpoint, cloneValues(form), c)
	outcome := s.Classify(response, raw, err)
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
// before the access token just minted is due for renewal: the next renewal will fail and
// only a reconnect helps. It names the connector and the connection, never a token.
func (s *Scheme) warnIfLastRefresh(p core.Profile, m material) {
	if m.RefreshExpiresAt.IsZero() || m.ExpiresAt.IsZero() || !m.RefreshExpiresAt.Before(m.ExpiresAt.Add(-margin(p))) {
		return
	}
	s.cfg.Logger.Warn("oauth2code: the refresh token expires before the next refresh; the connection will need a reconnect",
		"connector", p.ConnectorID, "connection", m.Ref.ConnectionID, "refresh_expires_at", m.RefreshExpiresAt)
}

// open reads the material this scheme sealed. Its errors never quote the payload.
func open(m core.Material) (material, error) {
	if m.Scheme != Name || m.Version != materialVersion {
		return material{}, fmt.Errorf("oauth2code: material is %q version %d, not %q version %d", m.Scheme, m.Version, Name, materialVersion)
	}
	var out material
	if json.Unmarshal(m.Payload, &out) != nil || out.AccessToken == "" {
		return material{}, errors.New("oauth2code: material payload is unreadable or has no access token")
	}
	return out, nil
}

// credential is the access token in m as a core.Credential, read back by Wrap.
func credential(m material) core.Credential {
	secret, _ := json.Marshal(map[string]string{"access_token": m.AccessToken})
	return core.NewCredential(Name, m.ExpiresAt, secret)
}

// margin is the manifest's refresh.margin, else defaultMargin.
func margin(p core.Profile) time.Duration {
	if p.Refresh.Margin > 0 {
		return time.Duration(p.Refresh.Margin)
	}
	return defaultMargin
}

// refreshExpiresAt is when a refresh token issued at now dies by the manifest's
// refresh.refresh_ttl, counted from each issue (a rotated token starts its own), or zero
// when the manifest does not say.
func refreshExpiresAt(p core.Profile, now time.Time) time.Time {
	if p.Refresh.RefreshTTL <= 0 {
		return time.Time{}
	}
	return now.Add(time.Duration(p.Refresh.RefreshTTL))
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
