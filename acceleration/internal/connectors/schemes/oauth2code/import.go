package oauth2code

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// The keys of core.CompleteInput.Supplied for a grant imported rather than consented to: a
// token response's members (RFC 6749 section 5.1), with expires_at in place of expires_in,
// since the grant was issued before it reaches the router.
const (
	SuppliedAccessToken  = "access_token"
	SuppliedRefreshToken = "refresh_token"
	// SuppliedExpiresAt is when the access token expires, in RFC 3339.
	SuppliedExpiresAt = "expires_at"
	// SuppliedScope is the granted scopes as the token response's scope says them: joined by
	// the manifest's scopes.separator, a space by default (RFC 6749 section 3.3).
	SuppliedScope = "scope"
)

// importGrant stores a grant the provider already issued, as the prototype's
// OAuthClientForImport did (internal/mcp/oauth.go:323-325 on codex/connector-support at
// cf62af0d): «It never accepts token or resource endpoints from the caller». The endpoints come
// from the manifest and the metadata it points to (discover), and the client from the app's or
// the operator's preregistered one (ClientLookup), never from Supplied: a refresh must go to the
// server, and with the client, the grant was issued to. A client the router would register now
// (cimd, dcr) is not that client, so it is never picked here.
//
// Nothing is captured: there is no callback and no token response, so the account id and
// metadata stay as the connection had them. The scopes are the ones Supplied names, each one
// the manifest lists.
func (s *Scheme) importGrant(ctx context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	allowed := []string{SuppliedAccessToken, SuppliedRefreshToken, SuppliedExpiresAt, SuppliedScope}
	for name := range in.Supplied {
		if !slices.Contains(allowed, name) {
			return core.StoredCredentials{}, core.AccountInfo{}, fmt.Errorf("oauth2code: %q is not a value an imported grant takes; it takes %s", name, strings.Join(allowed, ", "))
		}
	}
	access, refresh := in.Supplied[SuppliedAccessToken], in.Supplied[SuppliedRefreshToken]
	if access == "" {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: an imported grant needs access_token")
	}
	// RFC 6749 appendix A.12 and A.17: both tokens are 1*VSCHAR, so neither can break a header.
	if !vschar(access) || (refresh != "" && !vschar(refresh)) {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: access_token and refresh_token may hold only visible ASCII and spaces")
	}
	expires, err := time.Parse(time.RFC3339, in.Supplied[SuppliedExpiresAt])
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: an imported grant needs expires_at, an RFC 3339 time")
	}
	scopes, err := importedScopes(in.Manifest, in.Supplied[SuppliedScope])
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	server, err := s.discover(ctx, in.Manifest)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	c, err := s.pickClient(ctx, in.Ref, in.Manifest, server, "", preregistered)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	now := s.cfg.Now()
	stored := storedPayload{
		Ref:                in.Ref,
		Client:             c,
		TokenEndpoint:      server.Token,
		RevocationEndpoint: server.Revocation,
		Issuer:             server.Issuer,
		Resource:           server.Resource,
		AccessToken:        access,
		RefreshToken:       refresh,
		ExpiresAt:          expires.UTC(),
		Scopes:             scopes,
	}
	if refresh != "" {
		stored.RefreshExpiresAt = refreshExpiresAt(in.Manifest, now)
	}
	payload, err := json.Marshal(stored)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: payload}, core.AccountInfo{Scopes: scopes}, nil
}

// importedScopes is scope split by the manifest's separator. A manifest that lists scopes
// needs it, and each scope must be one it lists, as the prototype checked (putImportedOAuthGrant
// in internal/api/connectors.go at cf62af0d).
func importedScopes(m core.ResolvedManifest, scope string) ([]string, error) {
	var scopes []string
	for _, part := range strings.Split(scope, separator(m)) {
		if part = strings.TrimSpace(part); part != "" && !slices.Contains(scopes, part) {
			scopes = append(scopes, part)
		}
	}
	if len(m.Scopes.List) == 0 {
		return scopes, nil
	}
	if len(scopes) == 0 {
		return nil, errors.New("oauth2code: an imported grant of this connector needs scope, the scopes it was granted")
	}
	for _, granted := range scopes {
		if !slices.Contains(m.Scopes.List, granted) {
			return nil, fmt.Errorf("oauth2code: scope %q is not one this connector asks for", granted)
		}
	}
	return scopes, nil
}

// vschar is RFC 6749 appendix A's VSCHAR: %x20-7E.
func vschar(value string) bool {
	for i := range len(value) {
		if value[i] < 0x20 || value[i] > 0x7e {
			return false
		}
	}
	return true
}
