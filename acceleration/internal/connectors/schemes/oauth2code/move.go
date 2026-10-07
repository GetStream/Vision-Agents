package oauth2code

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// ErrGrantElsewhere is a grant this connector could not renew: it was issued at another token
// endpoint, or to a client the connector would not use. Moving it would only leave a
// connection that fails at its first refresh, so the person logs in again instead.
var ErrGrantElsewhere = errors.New("oauth2code: the grant belongs to another token endpoint or client")

// MovedGrant is a grant a plugin login holds (agent_plugin_connections): what the plugin
// system stored, which is less than a consent here learns. T61 in
// acceleration/docs/connectors/subtasks.md on connectors/planning.
type MovedGrant struct {
	AccessToken  string
	RefreshToken string
	// ExpiresAt is when the access token expires; zero when the provider said nothing, which
	// is how the plugin stored a token that does not expire (plugins.Auth.Exchange).
	ExpiresAt time.Time
	// ClientID is the client the grant was issued to, and TokenEndpoint where the plugin
	// renewed it.
	ClientID      string
	TokenEndpoint string
	// Scopes are what the plugin asked for: it never stored what was granted.
	Scopes []string
	// MaybeRegistered says the plugin may have registered ClientID itself (RFC 7591), which it
	// does only for a plugin that needs no client of its own, and only when the config set none
	// (plugins.Auth.StartAuthorize). Only then is a client the app does not have taken as that
	// registered public client.
	MaybeRegistered bool
}

// MoveGrant stores a plugin's grant as this scheme's credentials, without a consent. As an
// imported grant (import.go), the endpoints come from the manifest and its metadata, never from
// the grant: the token endpoint discovered must be the one the plugin renewed it at. The client
// must be the one the grant was issued to: a preregistered client of the connection's app
// (customer, managed, operator) with that id, else, when the plugin may have registered it and
// client.registration allows dcr, the public client the plugin registered (internal/plugins/oauth.go registers each with
// token_endpoint_auth_method none). Either mismatch is ErrGrantElsewhere.
//
// Nothing is registered and nothing is sent to the token endpoint. Nothing is captured, as for an
// imported grant.
func (s *Scheme) MoveGrant(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, g MovedGrant) (core.StoredCredentials, core.AccountInfo, error) {
	if g.AccessToken == "" || g.ClientID == "" {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: a moved grant needs an access token and its client")
	}
	// RFC 6749 appendix A.12 and A.17, as importGrant checks them.
	if !vschar(g.AccessToken) || !vschar(g.RefreshToken) {
		return core.StoredCredentials{}, core.AccountInfo{}, errors.New("oauth2code: access_token and refresh_token may hold only visible ASCII and spaces")
	}
	server, err := s.discover(ctx, m)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	if server.Token != g.TokenEndpoint {
		return core.StoredCredentials{}, core.AccountInfo{}, fmt.Errorf("%w: it was renewed at %s, and the connector's token endpoint is %s",
			ErrGrantElsewhere, g.TokenEndpoint, server.Token)
	}
	c, err := s.movedClient(ctx, ref, m, server, g)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	stored := storedPayload{
		Ref:                ref,
		Client:             c,
		TokenEndpoint:      server.Token,
		RevocationEndpoint: server.Revocation,
		Issuer:             server.Issuer,
		Resource:           server.Resource,
		AccessToken:        g.AccessToken,
		RefreshToken:       g.RefreshToken,
		ExpiresAt:          g.ExpiresAt.UTC(),
		Scopes:             slices.Clone(g.Scopes),
	}
	if g.RefreshToken != "" {
		stored.RefreshExpiresAt = refreshExpiresAt(m, s.cfg.Now())
	}
	payload, err := json.Marshal(stored)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: payload}, core.AccountInfo{Scopes: slices.Clone(g.Scopes)}, nil
}

// movedClient is the client a moved grant was issued to, as this scheme would hold it.
func (s *Scheme) movedClient(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, d server, g MovedGrant) (client, error) {
	id := g.ClientID
	var found []string
	for _, registration := range preregistered {
		if !slices.Contains(m.Client.Registration, registration) || s.cfg.Clients == nil {
			continue
		}
		c, ok, err := s.cfg.Clients(ctx, ref, m, registration)
		if err != nil {
			return client{}, fmt.Errorf("oauth2code: %s client: %w", registration, err)
		}
		if !ok {
			continue
		}
		if c.ID != id {
			found = append(found, fmt.Sprintf("%s %s", registration, c.ID))
			continue
		}
		method, err := preregisteredMethod(m, d, c)
		if err != nil {
			return client{}, err
		}
		return client{RegistrationMethod: registration, ID: c.ID, AuthMethod: method}, nil
	}
	if g.MaybeRegistered && slices.Contains(m.Client.Registration, core.ClientDCR) {
		if err := checkMethod(core.AuthNone, d); err != nil {
			return client{}, err
		}
		return client{RegistrationMethod: core.ClientDCR, ID: id, AuthMethod: core.AuthNone}, nil
	}
	return client{}, fmt.Errorf("%w: it was issued to client %s, and the connector would use %v (client.registration %v)",
		ErrGrantElsewhere, id, found, m.Client.Registration)
}
