//go:build integration

package resolver_test

import (
	"io/fs"
	"strings"
	"testing/fstest"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The built-in Salesforce manifest, seeded as the router seeds it with its endpoints pointed
// at the fake, and a connection the app owns: its client goes in once, no browser opens,
// and every later token comes from the same client, with no reconnect. The fake answers as
// Salesforce does, without expires_in, so the manifest's access_ttl (15 minutes) decides.
func (s *ResolverSuite) TestAnAppOwnedSalesforceConnectionResolvesATokenWithoutABrowser() {
	s.f.srv.Use(fakeprovider.ClientCredentials, fakeprovider.IdentityURL, fakeprovider.NoExpiresIn)
	ref := s.salesforceConnection()
	s.Equal(0, s.f.srv.Hits(fakeprovider.PathAuthorize), "no browser")
	s.Equal(1, s.f.srv.ClientCredentialsGrants(), "the client went in once, and its first token with it")
	s.Equal(s.f.srv.IdentityURL, s.f.stored(ref).AccountID)

	r := s.f.router(s.f.srv.Client())
	first, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.Equal(oauth2cc.Name, first.Scheme)
	s.True(s.f.works(first))
	s.Equal(1, s.f.srv.ClientCredentialsGrants(), "the stored token is handed out until it is due")
	s.True(first.ExpiresAt.Equal(s.f.clock.Now().Add(15*time.Minute)), "access_ttl, not the fake's hour")

	s.f.clock.Add(15 * time.Minute)
	renewed, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.token(first) != s.f.token(renewed), "a new token")
	s.True(s.f.works(renewed))
	s.Equal(2, s.f.srv.ClientCredentialsGrants())
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Empty(stored.LastError)
}

// A client credentials request spends nothing, so the scheme sets no checkpoint and a token
// endpoint that is down leaves the connection connected, unlike a lost refresh.
func (s *ResolverSuite) TestASalesforceTokenEndpointThatIsDownAfterExpiryKeepsTheConnection() {
	s.f.srv.Use(fakeprovider.ClientCredentials, fakeprovider.IdentityURL)
	ref := s.salesforceConnection()
	s.f.srv.Use(fakeprovider.ClientCredentials, fakeprovider.IdentityURL, fakeprovider.Unavailable)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrTemporarilyUnavailable)
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Equal(temporarilyUnavailable, stored.LastError)
}

// The follow-up to AI-846: after the client a grant was issued to is removed, the next
// refresh says so and the connection needs a reconnect, instead of «the client changed
// during the consent» on a connection that stays connected.
func (s *ResolverSuite) TestARefreshAfterTheClientWasRemovedMovesTheConnectionToNeedsReauthorization() {
	ref := s.f.connected()
	s.f.clientRemoved.Store(true)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
	var gone *oauth2code.ClientRemovedError
	s.Require().ErrorAs(err, &gone)
	s.ErrorContains(err, "OAuth client for this connector was removed; the connection needs a reconnect")
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(rejectedGrant, stored.LastError)
	s.Equal(0, s.f.srv.Refreshes(), "nothing was sent")
}

// salesforceConnection seeds providers/salesforce.yaml as a built-in with every endpoint at
// the fake, makes an app-owned oauth2_client_credentials connection from it, and connects it
// as the credentials endpoint (T18) will: Begin, which is Done, then Complete with the
// fake's client, stored under the lock.
func (s *ResolverSuite) salesforceConnection() core.ConnectionRef {
	raw, err := fs.ReadFile(providers.FS, "salesforce.yaml")
	s.Require().NoError(err)
	at := strings.NewReplacer(
		"https://{login_host}/services/oauth2/authorize", s.f.srv.URL+fakeprovider.PathAuthorize,
		"https://{login_host}/services/oauth2/token", s.f.srv.URL+fakeprovider.PathToken,
		"https://{login_host}/services/oauth2/revoke", s.f.srv.URL+fakeprovider.PathRevoke,
		"https://api.salesforce.com/{mcp_path}", s.f.srv.URL+fakeprovider.PathMCP,
	).Replace(string(raw))
	s.Require().NotContains(at, "{login_host}", "every endpoint is at the fake")
	s.Require().NoError(s.f.db.SeedConnectorDefinitions(s.f.ctx, fstest.MapFS{"salesforce.yaml": {Data: []byte(at)}}))
	definition, err := s.f.db.LatestConnectorDefinition(s.f.ctx, customer, "salesforce")
	s.Require().NoError(err)

	scheme := s.f.clientCredentials(s.f.srv.Client())
	connection := &store.ConnectorConnection{
		CustomerID: customer, ConnectorID: "salesforce", DefinitionRevision: definition.Revision,
		OwnerType: store.OwnerApp, AuthScheme: oauth2cc.Name,
	}
	registry := core.Registry{Schemes: map[string]core.Scheme{oauth2cc.Name: scheme}}
	s.Require().NoError(s.f.db.CreateConnectorConnection(s.f.ctx, registry, connection))
	ref := core.ConnectionRef{CustomerID: customer, ConnectionID: connection.ID}

	resolved, err := definition.Manifest.Resolve(oauth2cc.Name, connection.Inputs, nil)
	s.Require().NoError(err)
	out, err := scheme.Begin(s.f.ctx, core.BeginInput{Ref: ref, Manifest: resolved})
	s.Require().NoError(err)
	s.Require().True(out.Done)
	credentials, account, err := scheme.Complete(s.f.ctx, core.CompleteInput{Ref: ref, Manifest: resolved, Supplied: map[string]string{
		oauth2cc.SuppliedClientID: s.f.srv.ClientID, oauth2cc.SuppliedClientSecret: s.f.srv.ClientSecret,
	}})
	s.Require().NoError(err)
	s.f.update(ref, func(state *core.CredentialState) {
		state.Credentials = credentials
		state.Status = store.ConnectionConnected
		state.LastError = ""
		state.AccountID, state.Metadata, state.Scopes = account.AccountID, account.Metadata, account.Scopes
		state.ExpiresAt = time.Time{}
	})
	return ref
}
