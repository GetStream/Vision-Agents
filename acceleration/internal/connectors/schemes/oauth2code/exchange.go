package oauth2code

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// TokenError is an error the token endpoint answered with (RFC 6749 section 5.2).
type TokenError struct {
	Status int
	Code   string
}

func (e *TokenError) Error() string {
	return fmt.Sprintf("oauth2code: token endpoint refused the code (HTTP %d): %s", e.Status, e.Code)
}

// tokenResponse is the part of a token response (RFC 6749 section 5.1) the scheme reads
// itself. Everything else a provider adds is read by the manifest's capture rules from the
// raw body.
type tokenResponse struct {
	AccessToken  string
	TokenType    string
	RefreshToken string
	ExpiresIn    int64
	Scope        string
	HasScope     bool
	IDToken      string
}

// exchange redeems the code at the token endpoint (RFC 6749 section 4.1.3) and returns the
// parsed response and its raw body.
func (s *Scheme) exchange(ctx context.Context, p core.Profile, a attempt, c client, code string) (tokenResponse, json.RawMessage, error) {
	form := url.Values{}
	// The manifest's token parameters go first, so none of them replaces one below.
	for _, key := range slices.Sorted(maps.Keys(p.TokenParams)) {
		form.Set(key, p.TokenParams[key])
	}
	form.Set("grant_type", "authorization_code")
	form.Set("code", code)
	form.Set("redirect_uri", a.RedirectURI)
	// RFC 7636 section 4.5.
	form.Set("code_verifier", a.Verifier)
	if a.Resource != "" {
		// RFC 8707 section 2.2.
		form.Set("resource", a.Resource)
	}
	var basic bool
	switch c.AuthMethod {
	case core.AuthNone:
		// RFC 6749 section 4.1.3: client_id is REQUIRED when the client does not
		// authenticate.
		form.Set("client_id", c.ID)
	case core.AuthClientSecretPost:
		// RFC 6749 section 2.3.1, the body form.
		form.Set("client_id", c.ID)
		form.Set("client_secret", c.Secret)
	case core.AuthClientSecretBasic:
		basic = true
	default:
		return tokenResponse{}, nil, fmt.Errorf("oauth2code: client authentication %q is not implemented", c.AuthMethod)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, a.TokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return tokenResponse{}, nil, err
	}
	if basic {
		// RFC 6749 section 2.3.1: id and secret are each form-urlencoded (Appendix B) before
		// they become the Basic credentials, which SetBasicAuth alone does not do.
		request.SetBasicAuth(url.QueryEscape(c.ID), url.QueryEscape(c.Secret))
	}
	// RFC 6749 section 4.1.3: the body is application/x-www-form-urlencoded; section 5.1
	// answers JSON.
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set("Accept", "application/json")
	response, err := s.cfg.HTTP.Do(request)
	if err != nil {
		return tokenResponse{}, nil, fmt.Errorf("oauth2code: token: %w", err)
	}
	defer response.Body.Close()
	raw, err := read(response.Body)
	if err != nil {
		return tokenResponse{}, nil, err
	}
	body, err := decodeObject(raw)
	if err != nil {
		return tokenResponse{}, nil, fmt.Errorf("oauth2code: token: HTTP %d: %w", response.StatusCode, err)
	}
	// RFC 6749 section 5.2: an error member is an error, whatever the status: some
	// servers answer errors with 200, and a body with error is never a token.
	if code, ok := body["error"].(string); ok && code != "" {
		return tokenResponse{}, nil, &TokenError{Status: response.StatusCode, Code: code}
	}
	if response.StatusCode != http.StatusOK {
		// RFC 6749 section 5.1: success is 200 OK.
		return tokenResponse{}, nil, &TokenError{Status: response.StatusCode}
	}
	token, err := parseToken(body)
	if err != nil {
		return tokenResponse{}, nil, err
	}
	return token, raw, nil
}

// parseToken reads RFC 6749 section 5.1's members. access_token is REQUIRED; the others
// are kept when present.
func parseToken(body map[string]any) (tokenResponse, error) {
	var t tokenResponse
	var ok bool
	if t.AccessToken, ok = body["access_token"].(string); !ok || t.AccessToken == "" {
		return tokenResponse{}, errors.New("oauth2code: token response has no access_token")
	}
	t.TokenType, _ = body["token_type"].(string)
	t.RefreshToken, _ = body["refresh_token"].(string)
	t.IDToken, _ = body["id_token"].(string)
	if n, ok := body["expires_in"].(json.Number); ok {
		if seconds, err := n.Int64(); err == nil && seconds > 0 {
			t.ExpiresIn = seconds
		}
	}
	t.Scope, t.HasScope = body["scope"].(string)
	return t, nil
}

// scopes is what the server granted. RFC 6749 section 5.1: scope is OPTIONAL «if
// identical to the scope requested by the client», so without it the request stands.
func (t tokenResponse) scopes(p core.Profile, requested []string) []string {
	if !t.HasScope {
		return slices.Clone(requested)
	}
	var granted []string
	for _, scope := range strings.Split(t.Scope, separator(p)) {
		if scope = strings.TrimSpace(scope); scope != "" {
			granted = append(granted, scope)
		}
	}
	return granted
}

// expiresAt is when the access token expires: expires_in (RFC 6749 section 5.1), else the
// manifest's refresh.access_ttl, else never as far as the scheme knows.
func (t tokenResponse) expiresAt(p core.Profile, now time.Time) time.Time {
	switch {
	case t.ExpiresIn > 0:
		return now.Add(time.Duration(t.ExpiresIn) * time.Second)
	case p.Refresh.AccessTTL > 0:
		return now.Add(time.Duration(p.Refresh.AccessTTL))
	}
	return time.Time{}
}

// readsIDToken is whether a capture rule reads the id_token, which is when its claims are
// trusted and so checked.
func readsIDToken(p core.Profile) bool {
	return slices.ContainsFunc(p.Capture, func(rule core.CaptureRule) bool { return rule.From == core.FromIDToken })
}

// checkIDToken applies OpenID Connect Core 1.0 section 3.1.3.7 to an id_token from the
// token response. The signature is not checked: the token came «via direct communication
// between the Client and the Token Endpoint», so item 6 lets «the TLS server validation»
// stand in for it, and the egress client verified that server. The claims the rest of the
// section asks for are: iss equal to the issuer (item 2), aud containing this client's id
// (item 3), and exp still ahead (item 9).
func checkIDToken(t tokenResponse, issuer, clientID string, now time.Time) error {
	if t.IDToken == "" {
		return errors.New("oauth2code: the manifest reads the id_token and the token response has none")
	}
	segments := strings.Split(t.IDToken, ".")
	if len(segments) != 3 {
		return errors.New("oauth2code: id_token is not a compact JWS of three segments")
	}
	payload, err := base64.RawURLEncoding.DecodeString(segments[1])
	if err != nil {
		return fmt.Errorf("oauth2code: id_token payload: %w", err)
	}
	claims, err := decodeObject(payload)
	if err != nil {
		return fmt.Errorf("oauth2code: id_token payload: %w", err)
	}
	if iss, _ := claims["iss"].(string); iss != issuer {
		return fmt.Errorf("oauth2code: id_token iss %q is not the issuer %q", iss, issuer)
	}
	var audience []string
	switch aud := claims["aud"].(type) {
	case string:
		audience = []string{aud}
	case []any:
		for _, a := range aud {
			if s, ok := a.(string); ok {
				audience = append(audience, s)
			}
		}
	}
	if !slices.Contains(audience, clientID) {
		return errors.New("oauth2code: id_token is not for this client")
	}
	exp, ok := claims["exp"].(json.Number)
	if !ok {
		return errors.New("oauth2code: id_token has no exp")
	}
	seconds, err := exp.Int64()
	if err != nil || !now.Before(time.Unix(seconds, 0)) {
		return errors.New("oauth2code: id_token has expired")
	}
	return nil
}

// decodeObject reads a JSON object, keeping numbers as written so an id is not rounded.
func decodeObject(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var object map[string]any
	if err := decoder.Decode(&object); err != nil {
		return nil, fmt.Errorf("not a JSON object: %w", err)
	}
	return object, nil
}
