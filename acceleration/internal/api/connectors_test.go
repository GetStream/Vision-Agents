//go:build integration

package api

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectorsSuite covers the MCP connector catalog, the accounts an app connects through
// it, and the logins that connect them.
//
// A connector's OAuth provider is a server answering in process. The router may reach one
// because the suite hands it a client that does not insist on a public HTTPS host; what a
// real provider makes of a login is that provider's business, and what is under test here
// is everything the router decides around it.
type ConnectorsSuite struct {
	RouterSuite
}

func TestConnectorsSuite(t *testing.T) {
	runSuite(t, new(ConnectorsSuite))
}

func (s *ConnectorsSuite) SetupSuite() {
	s.oauthHTTP = &http.Client{Timeout: settleFor}
	s.RouterSuite.SetupSuite()
}

// SetupTest gives every test an app of its own, because a list of connections is everything
// an app has, and a connector an app registers is named by an id only that app reserves.
func (s *ConnectorsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectorsSuite) TestEveryConnectorInTheCatalogIsListed() {
	// A built-in catalog rather than the app's own rows, so a fresh app has all of it.
	listed := s.catalog("")

	s.Len(listed, 7)
	s.Equal("slack", listed[0].Id)
}

func (s *ConnectorsSuite) TestSearchingTheCatalogFindsTheConnectorsThatMatch() {
	listed := s.catalog("cal")

	s.Equal([]string{"calendly", "calcom"}, named(listed))
}

func (s *ConnectorsSuite) TestTheCatalogIsFilteredByWhatAConnectorIsFor() {
	scheduling := s.catalog("Scheduling")

	s.Require().NotEmpty(scheduling)
	for _, connector := range scheduling {
		s.Equal("Scheduling", connector.Category)
	}
	s.NotContains(named(scheduling), "slack")
}

func (s *ConnectorsSuite) TestAConnectorAnAppRegisteredIsListedForThatAppAlone() {
	var created ConnectorDefinition
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connectors",
		map[string]any{"id": "custom_crm", "name": "Custom CRM", "endpoint": "https://8.8.8.8/mcp", "auth_mode": "none"},
		&created))
	s.Contains(named(s.catalog("")), "custom_crm")

	stranger := s.data.backendOfAnotherApp()
	var theirs []ConnectorDefinition
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/agents/connectors", nil, &theirs))
	s.NotContains(named(theirs), "custom_crm")
	s.Equal(http.StatusNotFound, stranger.do(http.MethodGet, "/v1/agents/connectors/custom_crm", nil, nil))
}

func (s *ConnectorsSuite) TestAConnectorThatIsNotInTheCatalogIsNotFound() {
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/agents/connectors/carrier-pigeon", nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, unknownConnector)
}

func (s *ConnectorsSuite) TestConnectingSomethingThatIsNotAConnectorIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections",
		map[string]any{"connector_id": "carrier-pigeon", "owner": map[string]any{"type": "app"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, unknownConnector)
}

func (s *ConnectorsSuite) TestASalesforceConnectionNamingAnInstanceItDoesNotHaveIsRefused() {
	// Salesforce has no single global host, and the instance is what picks one.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections", map[string]any{
		"connector_id": "salesforce", "owner": map[string]any{"type": "app"}, "instance": "elsewhere",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "production or sandbox")
}

func (s *ConnectorsSuite) TestAnAppHoldsNoConnectionsUntilOneIsMade() {
	var listed []ConnectorConnection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections", nil, &listed))

	s.Empty(listed)
}

func (s *ConnectorsSuite) TestAConnectionNobodyMadeIsNotFound() {
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/agents/connections/"+s.utils.uuid(), nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, "no such connection")
}

func (s *ConnectorsSuite) TestDisconnectingAConnectionNobodyMadeIsNotFound() {
	s.Equal(http.StatusNotFound,
		s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+s.utils.uuid(), nil, nil))
}

func (s *ConnectorsSuite) TestLoggingIntoAConnectionNobodyMadeIsNotFound() {
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodPost,
		"/v1/agents/connections/"+s.utils.uuid()+"/authorizations", map[string]any{}, nil))
}

func (s *ConnectorsSuite) TestLoggingIntoSlackWithoutChoosingItsScopesIsRefused() {
	// Refused before the login starts, so nothing reaches Slack asking for every scope it has.
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/connections/"+connection.ID+"/authorizations", map[string]any{})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "minimum Slack scopes")
}

func (s *ConnectorsSuite) TestAnotherAppsConnectionIsNotFound() {
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+connection.ID, nil, nil)
	})
}

func (s *ConnectorsSuite) TestAnotherAppListsNoneOfThisAppsConnections() {
	s.seedConnection("slack", "https://mcp.slack.com/mcp")

	var theirs []ConnectorConnection
	s.Require().Equal(http.StatusOK,
		s.data.backendOfAnotherApp().do(http.MethodGet, "/v1/agents/connections", nil, &theirs))

	s.Empty(theirs)
}

func (s *ConnectorsSuite) TestOnlyTheAppsOwnBackendMayReadTheCatalog() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connectors", nil, nil)
	})
}

func (s *ConnectorsSuite) TestOnlyTheAppsOwnBackendMayListConnections() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections", nil, nil)
	})
}

func (s *ConnectorsSuite) TestAnEndUsersDeviceMayNotStartALogin() {
	// Refused before the login is started, so nothing reaches the provider on behalf of
	// somebody who may not connect an account.
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")

	s.Equal(http.StatusForbidden, s.client.do(http.MethodPost,
		"/v1/agents/connections/"+connection.ID+"/authorizations",
		map[string]any{"scopes": []string{"search:read.public"}}, nil))
}

func (s *ConnectorsSuite) TestTheProviderArrivesAtTheCallbackWithoutACredential() {
	// The browser comes from the identity provider holding none of ours, and the state is
	// the secret. A 401 here would be a login that can never finish.
	status, body := s.unauthenticatedClient.call(http.MethodGet, mcp.CallbackPath, nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "authorization state is invalid")
}

func (s *ConnectorsSuite) TestALoginNobodyStartedIsRefused() {
	status, body := s.unauthenticatedClient.call(http.MethodGet,
		mcp.CallbackPath+"?state="+s.utils.uuid()+"&code="+s.utils.uuid(), nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "expired or already used")
}

func (s *ConnectorsSuite) TestTheLoginLaunchPageIsIsolatedAndTargetsTheConfiguredDashboard() {
	response, body := s.browse(s.request(http.MethodGet, connectorOAuthLaunchPath+s.utils.uuid(), "", ""))

	s.Equal(http.StatusOK, response.StatusCode)
	s.Contains(response.Header.Get("Content-Security-Policy"), "script-src 'nonce-")
	s.Equal("no-store", response.Header.Get("Cache-Control"))
	s.Contains(string(body), `"https://dashboard.test"`, "only the dashboard's origin may hand the login over")
	s.Contains(string(body), "va.connector.oauth.ready")
}

func (s *ConnectorsSuite) TestTheOAuthClientMetadataIsPublicAndMatchesTheCallback() {
	response, body := s.browse(s.request(http.MethodGet, mcp.ClientMetadataPath, "", ""))

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("application/json", response.Header.Get("Content-Type"))
	s.Equal("public, max-age=300", response.Header.Get("Cache-Control"))
	s.Equal("nosniff", response.Header.Get("X-Content-Type-Options"))
	var metadata struct {
		ClientID                string   `json:"client_id"`
		RedirectURIs            []string `json:"redirect_uris"`
		TokenEndpointAuthMethod string   `json:"token_endpoint_auth_method"`
	}
	s.Require().NoError(json.Unmarshal(body, &metadata))
	s.Equal(s.server.URL+mcp.ClientMetadataPath, metadata.ClientID)
	s.Equal([]string{s.server.URL + mcp.CallbackPath}, metadata.RedirectURIs)
	s.Equal("none", metadata.TokenEndpointAuthMethod)
}

func (s *ConnectorsSuite) TestACustomAPIKeyConnectorKeepsItsCredentialWriteOnlyAndUsesItAtRuntime() {
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connectors", map[string]any{
		"id": "custom_crm", "name": "Custom CRM", "endpoint": "https://8.8.8.8/mcp",
		"auth_mode": "api_key", "api_key_header": "X-Api-Key",
	}, nil))

	var connection ConnectorConnection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", map[string]any{
		"connector_id": "custom_crm", "owner": map[string]any{"type": "app"}, "label": "Sales CRM",
	}, &connection))
	s.Equal(ConnectorConnectionAuthTypeApiKey, connection.AuthType)
	s.Equal(ConnectorConnectionStatusPending, connection.Status)
	s.Equal(1, connection.Revision)

	secret := "customer-api-key-" + s.utils.uuid()
	status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+connection.Id+"/credentials",
		map[string]any{"expected_revision": 1, "api_key": secret})
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.NotContains(string(payload), secret, "a credential is written and never read back")
	var connected ConnectorConnection
	s.Require().NoError(json.Unmarshal(payload, &connected))
	s.Equal(ConnectorConnectionStatusConnected, connected.Status)
	s.Equal(2, connected.Revision)

	stored := s.storedConnection(connection.Id)
	s.NotContains(string(stored.CredentialSealed), secret)
	request := httptest.NewRequest(http.MethodPost, stored.Endpoint, nil)
	s.Require().NoError(connectors.AuthorizeRequest(context.Background(), s.store, s.sealer,
		s.customerID(), connection.Id, nil, request))
	s.Equal(secret, request.Header.Get("X-Api-Key"))
}

func (s *ConnectorsSuite) TestAConnectorCredentialIsRewrappedToTheCurrentKeyOnUse() {
	oldSealer, err := auth.NewSealer("old-connector-kek")
	s.Require().NoError(err)
	rotatedSealer, err := auth.NewSealerWithKeyring(2, map[int]string{
		1: "old-connector-kek",
		2: "current-connector-kek",
	})
	s.Require().NoError(err)

	connection := s.seedConnection("github", "https://api.githubcopilot.com/mcp/")
	expiresAt := time.Now().UTC().Add(time.Hour)
	connection.ExpiresAt = &expiresAt
	connection = s.connectWith(oldSealer, connection,
		connectors.Credentials{AuthType: connectors.AuthOAuth2, AccessToken: "pre-rotation-access-token"})

	resolved, err := connectors.ResolveCredentials(context.Background(), s.store, rotatedSealer,
		s.customerID(), connection.ID, nil)
	s.Require().NoError(err)
	s.Equal("pre-rotation-access-token", resolved.AccessToken)

	stored := s.storedConnection(connection.ID)
	s.Equal(2, stored.Revision, "rotating the wrapping key does not change the grant's revision")
	s.Equal(2, stored.CredentialKEKVersion)
	opened, err := connectors.OpenCredentials(rotatedSealer, s.customerID(), connection.ID, stored.Revision,
		stored.CredentialKEKVersion, stored.CredentialSealed)
	s.Require().NoError(err)
	s.Equal("pre-rotation-access-token", opened.AccessToken)
}

func (s *ConnectorsSuite) TestAUserCannotReadValidateOrDisconnectAnotherUsersConnection() {
	alice, bob := s.data.createUser(), s.data.createUser()
	connection := s.seedOwnedConnection(alice.userID)
	connection.Status = store.ConnectorConnected
	connection.CredentialSealed = []byte("sealed-user-grant")
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(context.Background(), &connection, connection.Revision))

	for _, as := range []*testClient{s.serverClient, s.serverClient.actingFor(bob)} {
		var listed []ConnectorConnection
		s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections", nil, &listed))
		s.Empty(listed, "a user's connection is listed only to the backend acting for them")

		s.Equal(http.StatusNotFound, as.do(http.MethodGet, "/v1/agents/connections/"+connection.ID, nil, nil))
		s.Equal(http.StatusNotFound, as.do(http.MethodGet, "/v1/agents/connections/"+connection.ID+"/tools", nil, nil))
		s.Equal(http.StatusNotFound, as.do(http.MethodPost, "/v1/agents/connections/"+connection.ID+"/validate", nil, nil))
		s.Equal(http.StatusNotFound, as.do(http.MethodDelete, "/v1/agents/connections/"+connection.ID, nil, nil))
	}
	s.Equal(http.StatusOK, s.serverClient.actingFor(alice).do(http.MethodGet,
		"/v1/agents/connections/"+connection.ID, nil, nil), "the backend acting for its owner reaches it")

	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal([]byte("sealed-user-grant"), stored.CredentialSealed)
}

func (s *ConnectorsSuite) TestABackendMayCreateAConnectionForTheUserItActsFor() {
	alice := s.data.createUser()
	s.seedDefinition("custom_identity", "https://8.8.8.8/mcp", connectors.AuthNone)

	var created ConnectorConnection
	s.Require().Equal(http.StatusCreated, s.serverClient.actingFor(alice).do(http.MethodPost,
		"/v1/agents/connections", s.ownedBy("custom_identity", alice.userID), &created))

	s.Equal(ConnectorConnectionOwnerTypeUser, created.OwnerType)
	s.Equal(alice.userID, value(created.OwnerId))
}

func (s *ConnectorsSuite) TestABackendCannotMakeAnotherUserTheOwnerOfAConnection() {
	alice, bob := s.data.createUser(), s.data.createUser()
	s.seedDefinition("custom_identity", "https://8.8.8.8/mcp", connectors.AuthNone)

	status, failure := s.serverClient.actingFor(alice).failure(http.MethodPost,
		"/v1/agents/connections", s.ownedBy("custom_identity", bob.userID))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "must match the verified user")
}

func (s *ConnectorsSuite) TestABackendActingForNobodyCannotCreateAUsersConnection() {
	alice := s.data.createUser()
	s.seedDefinition("custom_identity", "https://8.8.8.8/mcp", connectors.AuthNone)

	status, failure := s.serverClient.failure(http.MethodPost,
		"/v1/agents/connections", s.ownedBy("custom_identity", alice.userID))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "must match the verified user")
}

func (s *ConnectorsSuite) TestAnEndUsersDeviceCannotCreateAConnectionForItself() {
	s.seedDefinition("custom_identity", "https://8.8.8.8/mcp", connectors.AuthNone)

	s.Equal(http.StatusForbidden, s.client.do(http.MethodPost,
		"/v1/agents/connections", s.ownedBy("custom_identity", s.client.userID), nil))
}

func (s *ConnectorsSuite) TestAnImportedOAuthGrantIsStoredEncryptedAndBoundToItsProvider() {
	var provider *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"resource": provider.URL + "/resource", "authorization_servers": []string{provider.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                provider.URL,
			"authorization_endpoint":                provider.URL + "/authorize",
			"token_endpoint":                        provider.URL + "/token",
			"jwks_uri":                              provider.URL + "/jwks",
			"response_types_supported":              []string{"code"},
			"code_challenge_methods_supported":      []string{"S256"},
			"token_endpoint_auth_methods_supported": []string{"client_secret_basic"},
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		clientID, clientSecret, ok := r.BasicAuth()
		if !ok || clientID != "imported-public-client" || clientSecret != "imported-client-secret" ||
			r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "imported-refresh-token" ||
			r.FormValue("resource") != provider.URL+"/resource" {
			http.Error(w, "unexpected token refresh request", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "expires_in": 3600,
		})
	})
	provider = httptest.NewServer(mux)
	s.T().Cleanup(provider.Close)

	s.seedDefinition("custom_import", provider.URL+"/mcp", connectors.AuthOAuth2)
	connection := s.seedConnection("custom_import", provider.URL+"/mcp")

	const accessToken = "imported-access-token"
	const refreshToken = "imported-refresh-token"
	const clientSecret = "imported-client-secret"
	status, payload := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+connection.ID+"/credentials",
		map[string]any{
			"expected_revision":   1,
			"access_token":        accessToken,
			"refresh_token":       refreshToken,
			"expires_at":          time.Now().UTC().Add(-time.Minute),
			"granted_scopes":      []string{},
			"oauth_client_id":     "imported-public-client",
			"oauth_client_secret": clientSecret,
		})
	s.Require().Equal(http.StatusOK, status, string(payload))
	s.NotContains(string(payload), accessToken)
	s.NotContains(string(payload), refreshToken)
	s.NotContains(string(payload), clientSecret)

	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(2, stored.Revision)
	s.NotContains(string(stored.CredentialSealed), accessToken)
	s.NotContains(string(stored.CredentialSealed), refreshToken)
	s.NotContains(string(stored.CredentialSealed), clientSecret)
	credentials := s.openCredentials(stored)
	s.Equal(accessToken, credentials.AccessToken)
	s.Equal(refreshToken, credentials.RefreshToken)
	s.Equal("imported-public-client", credentials.OAuthClientID)
	s.Equal(clientSecret, credentials.OAuthClientSecret)
	s.Equal("client_secret_basic", credentials.ClientAuthMethod)
	s.Equal(provider.URL, credentials.OAuthIssuer)
	s.Equal(provider.URL+"/resource", credentials.Resource)
	s.Equal(provider.URL+"/token", credentials.TokenEndpoint)

	// The grant was imported already expired, so using it is what redeems the refresh token.
	request := httptest.NewRequest(http.MethodPost, stored.Endpoint, nil)
	s.Require().NoError(connectors.AuthorizeRequest(context.Background(), s.store, s.sealer, s.customerID(),
		connection.ID, &mcp.OAuthClient{HTTP: provider.Client()}, request))
	s.Equal("Bearer rotated-access-token", request.Header.Get("Authorization"))
	stored = s.storedConnection(connection.ID)
	s.Equal(3, stored.Revision)
	credentials = s.openCredentials(stored)
	s.Equal("rotated-access-token", credentials.AccessToken)
	s.Equal("rotated-refresh-token", credentials.RefreshToken)
}

func (s *ConnectorsSuite) TestAnOAuthLoginExchangesItsPKCECodeAndStoresTheGrantAfterTheBrowserHandoff() {
	var provider *httptest.Server
	var registered, exchanged atomic.Bool
	var tokenRequests atomic.Int32
	var challenge atomic.Value
	challenge.Store("")
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"resource": provider.URL + "/resource", "authorization_servers": []string{provider.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"issuer":                                         provider.URL,
			"authorization_endpoint":                         provider.URL + "/authorize",
			"token_endpoint":                                 provider.URL + "/token",
			"registration_endpoint":                          provider.URL + "/register",
			"response_types_supported":                       []string{"code"},
			"code_challenge_methods_supported":               []string{"S256"},
			"token_endpoint_auth_methods_supported":          []string{"none"},
			"authorization_response_iss_parameter_supported": true,
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, "invalid registration request", http.StatusBadRequest)
			return
		}
		registered.Store(r.Method == http.MethodPost && request["token_endpoint_auth_method"] == "none")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"client_id": "connector-e2e-client", "token_endpoint_auth_method": "none",
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		tokenRequests.Add(1)
		_ = r.ParseForm()
		digest := sha256.Sum256([]byte(r.FormValue("code_verifier")))
		exchanged.Store(r.Method == http.MethodPost && r.FormValue("grant_type") == "authorization_code" &&
			r.FormValue("code") == "accepted-code" && r.FormValue("client_id") == "connector-e2e-client" &&
			r.FormValue("redirect_uri") == s.server.URL+mcp.CallbackPath &&
			base64.RawURLEncoding.EncodeToString(digest[:]) == challenge.Load().(string) &&
			r.FormValue("resource") == provider.URL+"/resource")
		if !exchanged.Load() {
			http.Error(w, "invalid token request", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "callback-access-token", "refresh_token": "callback-refresh-token",
			"expires_in": 3600, "scope": "read:workspace",
		})
	})
	provider = httptest.NewServer(mux)
	s.T().Cleanup(provider.Close)

	s.seedDefinition("custom_oauth_callback", provider.URL+"/mcp", connectors.AuthOAuth2)
	connection := s.seedConnection("custom_oauth_callback", provider.URL+"/mcp")

	var authorization ConnectorAuthorization
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/connections/"+connection.ID+"/authorizations", map[string]any{}, &authorization))
	launch, err := url.Parse(authorization.AuthorizationUrl)
	s.Require().NoError(err)
	s.Equal(s.server.URL, launch.Scheme+"://"+launch.Host, "the browser is sent to the router before the provider")

	handoff := s.request(http.MethodPost, launch.Path, s.server.URL, `{"handoff_token":"`+authorization.HandoffToken+`"}`)
	response, body := s.browse(handoff)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(body))
	var handedOff struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	s.Require().NoError(json.Unmarshal(body, &handedOff))
	cookies := response.Cookies()
	s.Require().Len(cookies, 1)

	authorize, err := url.Parse(handedOff.AuthorizationURL)
	s.Require().NoError(err)
	state := authorize.Query().Get("state")
	s.NotEmpty(state)
	s.Equal("S256", authorize.Query().Get("code_challenge_method"))
	s.Equal("connector-e2e-client", authorize.Query().Get("client_id"))
	s.Require().NotEmpty(authorize.Query().Get("code_challenge"))
	challenge.Store(authorize.Query().Get("code_challenge"))

	returned := url.Values{"state": {state}, "code": {"accepted-code"}, "iss": {provider.URL}}
	finished := s.arrive(returned, cookies[0])
	s.Equal(http.StatusFound, finished.StatusCode)
	s.Equal(connection.ID, s.dashboardReturn(finished).Get("connection_id"))
	s.Equal("connected", s.dashboardReturn(finished).Get("status"))
	s.True(registered.Load())
	s.True(exchanged.Load(), "the code is exchanged with the PKCE verifier and the expected resource")
	s.Equal(int32(1), tokenRequests.Load())

	replayed := s.arrive(returned, cookies[0])
	s.Equal(http.StatusBadRequest, replayed.StatusCode, "a login that finished cannot be finished twice")
	s.Equal(int32(1), tokenRequests.Load(), "a replay does not redeem the provider's code again")

	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(2, stored.Revision)
	s.NotContains(string(stored.CredentialSealed), "callback-access-token")
	credentials := s.openCredentials(stored)
	s.Equal("callback-access-token", credentials.AccessToken)
	s.Equal("callback-refresh-token", credentials.RefreshToken)
	s.Equal("connector-e2e-client", credentials.OAuthClientID)
	s.Equal(provider.URL, credentials.OAuthIssuer)
}

func (s *ConnectorsSuite) TestConcurrentUsesOfAnExpiredGrantRedeemItsRefreshTokenOnce() {
	var refreshes atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil || r.Method != http.MethodPost ||
			r.FormValue("grant_type") != "refresh_token" || r.FormValue("refresh_token") != "old-refresh-token" {
			http.Error(w, "unexpected refresh request", http.StatusBadRequest)
			return
		}
		refreshes.Add(1)
		time.Sleep(20 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "rotated-access-token", "refresh_token": "rotated-refresh-token", "expires_in": 3600,
		})
	}))
	s.T().Cleanup(provider.Close)

	connection := s.seedConnection("github", "https://api.githubcopilot.com/mcp/")
	expiredAt := time.Now().UTC().Add(-time.Minute)
	connection.ExpiresAt = &expiredAt
	connection = s.connectWith(s.sealer, connection, connectors.Credentials{
		AuthType:        connectors.AuthOAuth2,
		AccessToken:     "expired-access-token",
		RefreshToken:    "old-refresh-token",
		OAuthClientID:   "github-test-client",
		TokenEndpoint:   provider.URL + "/token",
		RefreshEndpoint: provider.URL + "/token",
	})

	const callers = 8
	type result struct {
		credentials connectors.Credentials
		err         error
	}
	start := make(chan struct{})
	results := make(chan result, callers)
	var wait sync.WaitGroup
	authenticator := &mcp.OAuthClient{HTTP: provider.Client()}
	for range callers {
		wait.Go(func() {
			<-start
			credentials, err := connectors.ResolveCredentials(context.Background(), s.store, s.sealer,
				s.customerID(), connection.ID, authenticator)
			results <- result{credentials: credentials, err: err}
		})
	}
	close(start)
	wait.Wait()
	close(results)
	for got := range results {
		s.Require().NoError(got.err)
		s.Equal("rotated-access-token", got.credentials.AccessToken)
		s.Equal("rotated-refresh-token", got.credentials.RefreshToken)
	}
	s.Equal(int32(1), refreshes.Load(), "a rotating refresh token is redeemed once")

	stored := s.storedConnection(connection.ID)
	s.Equal(connection.Revision+1, stored.Revision)
	s.Equal(store.ConnectorConnected, stored.Status)
	credentials := s.openCredentials(stored)
	s.Equal("rotated-access-token", credentials.AccessToken)
	s.Equal("rotated-refresh-token", credentials.RefreshToken)
}

func (s *ConnectorsSuite) TestARefreshWhoseResponseIsLostNeedsReauthorization() {
	s.assertRefreshIsUncertain(func(w http.ResponseWriter, _ context.CancelFunc) {
		_, _ = w.Write([]byte(`{"access_token":`))
	})
}

func (s *ConnectorsSuite) TestARefreshWhoseWorkerIsCanceledNeedsReauthorization() {
	s.assertRefreshIsUncertain(func(_ http.ResponseWriter, cancel context.CancelFunc) {
		cancel()
	})
}

func (s *ConnectorsSuite) TestATemporaryRefreshFailureBeforeExpiryKeepsUsingTheCurrentToken() {
	refresh := s.refreshAgainst(time.Now().Add(30*time.Second), temporarilyUnavailable)

	s.Require().NoError(refresh.err)
	s.Equal("old-access", refresh.credentials.AccessToken)
	s.Equal(store.ConnectorConnected, refresh.stored.Status)
	s.Contains(refresh.stored.LastError, "temporarily unavailable")
}

func (s *ConnectorsSuite) TestATemporaryRefreshFailureAfterExpiryIsReportedAsTemporary() {
	refresh := s.refreshAgainst(time.Now().Add(-time.Minute), temporarilyUnavailable)

	s.ErrorIs(refresh.err, connectors.ErrCredentialTemporarilyUnavailable)
	s.Equal(store.ConnectorConnected, refresh.stored.Status)
	s.Contains(refresh.stored.LastError, "temporarily unavailable")
}

func (s *ConnectorsSuite) TestAProviderDenialPreservesTheExistingGrant() {
	connection := s.connectWith(s.sealer, s.seedConnection("slack", "https://mcp.slack.com/mcp"),
		connectors.Credentials{
			AuthType:     connectors.AuthOAuth2,
			AccessToken:  "existing-slack-access-token",
			RefreshToken: "existing-slack-refresh-token",
		})
	login := s.startedLogin(connection, mcp.PendingAuthorization{})

	response := s.arrive(url.Values{"state": {login.state}, "error": {"access_denied"}}, login.cookie())

	s.Equal(http.StatusFound, response.StatusCode)
	s.Equal("failed", s.dashboardReturn(response).Get("status"))
	_, err := s.store.ConnectorAuthorizationAttemptByState(context.Background(), login.state)
	s.Error(err, "a provider's denial consumes the login")
	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(connection.Revision, stored.Revision)
	s.Equal(connection.CredentialSealed, stored.CredentialSealed)
	credentials := s.openCredentials(stored)
	s.Equal("existing-slack-access-token", credentials.AccessToken)
	s.Equal("existing-slack-refresh-token", credentials.RefreshToken)
}

func (s *ConnectorsSuite) TestALoginThatSelectsAnotherAccountPreservesTheConnection() {
	var account atomic.Value
	account.Store([2]string{"T-old", "U-old"})
	var tokenRequests atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests.Add(1)
		selected := account.Load().([2]string)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  "new-account-access-token",
			"refresh_token": "new-account-refresh-token",
			"team":          map[string]string{"id": selected[0]},
			"authed_user":   map[string]string{"id": selected[1]},
		})
	}))
	s.T().Cleanup(provider.Close)
	slack := func() mcp.PendingAuthorization {
		return mcp.PendingAuthorization{
			ConnectorID: "slack", ClientID: "slack-test-client", ClientAuthMethod: "none", TokenEndpoint: provider.URL,
		}
	}
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")

	first := s.startedLogin(connection, slack())
	response := s.arrive(url.Values{"state": {first.state}, "code": {"provider-code"}}, first.cookie())
	s.Equal(http.StatusFound, response.StatusCode)
	s.Equal("connected", s.dashboardReturn(response).Get("status"))
	connection = s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, connection.Status)
	s.Equal("slack:T-old:U-old", connection.AccountID, "the callback keeps the provider's stable identity")

	account.Store([2]string{"T-new", "U-new"})
	second := s.startedLogin(connection, slack())
	response = s.arrive(url.Values{"state": {second.state}, "code": {"provider-code"}}, second.cookie())
	s.Equal(http.StatusFound, response.StatusCode)
	s.Equal("failed", s.dashboardReturn(response).Get("status"))
	s.Equal(int32(2), tokenRequests.Load())

	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(connection.Revision, stored.Revision)
	s.Equal("slack:T-old:U-old", stored.AccountID)
	s.Contains(stored.LastError, "different or unverified provider account")
	credentials := s.openCredentials(stored)
	s.Equal("new-account-access-token", credentials.AccessToken)
	s.Equal("new-account-refresh-token", credentials.RefreshToken)
}

func (s *ConnectorsSuite) TestALoginMustFinishInTheBrowserThatStartedItBeforeItsStateIsConsumed() {
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")
	login := s.startedLogin(connection, mcp.PendingAuthorization{CodeVerifier: "pkce-verifier"})
	returned := url.Values{"state": {login.state}}

	s.Equal(http.StatusForbidden, s.arrive(returned, nil).StatusCode)
	s.Equal(http.StatusForbidden, s.arrive(returned,
		&http.Cookie{Name: connectorAuthorizationCookieName(login.attemptID), Value: "wrong-browser"}).StatusCode)
	_, err := s.store.ConnectorAuthorizationAttemptByState(context.Background(), login.state)
	s.Require().NoError(err, "a browser that is refused does not spend the real user's one-time state")

	s.Equal(http.StatusBadRequest, s.arrive(returned, login.cookie()).StatusCode,
		"the browser that started it gets as far as the missing code")
	_, err = s.store.ConnectorAuthorizationAttemptByState(context.Background(), login.state)
	s.Error(err, "the browser that started it spends the state exactly once")
}

func (s *ConnectorsSuite) TestALoginReturningFromAnUnexpectedAuthorizationServerIsRefused() {
	var tokenRequests atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"access_token": "unexpected"})
	}))
	s.T().Cleanup(provider.Close)
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")
	login := s.startedLogin(connection, mcp.PendingAuthorization{
		CodeVerifier:  "pkce-verifier",
		ClientID:      "oauth-client",
		Issuer:        "https://expected.example",
		RequireIssuer: true,
		TokenEndpoint: provider.URL,
	})

	response := s.arrive(url.Values{
		"state": {login.state}, "code": {"provider-code"}, "iss": {"https://attacker.example"},
	}, login.cookie())

	s.Equal(http.StatusFound, response.StatusCode)
	s.Equal("failed", s.dashboardReturn(response).Get("status"))
	s.Zero(tokenRequests.Load(), "a mismatched issuer is refused before the code is redeemed")
	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorFailed, stored.Status)
	s.Equal("authorization server identity could not be verified", stored.LastError)
	s.Empty(stored.CredentialSealed)
}

func (s *ConnectorsSuite) TestALoginStartedForAReplacedConnectionRevisionIsRefused() {
	var tokenRequests atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		tokenRequests.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": "stale-authorization-token", "refresh_token": "stale-refresh-token",
		})
	}))
	s.T().Cleanup(provider.Close)
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")
	login := s.startedLogin(connection, mcp.PendingAuthorization{
		CodeVerifier:     "stale-pkce-verifier",
		ClientID:         "oauth-client",
		ClientAuthMethod: "none",
		TokenEndpoint:    provider.URL,
	})
	replacement := s.connectWith(s.sealer, connection,
		connectors.Credentials{AuthType: connectors.AuthOAuth2, AccessToken: "replacement-account-token"})

	response := s.arrive(url.Values{"state": {login.state}, "code": {"provider-code"}}, login.cookie())

	s.Equal(http.StatusConflict, response.StatusCode)
	s.Zero(tokenRequests.Load(), "a login for the replaced revision does not redeem its code")
	_, err := s.store.ConnectorAuthorizationAttemptByState(context.Background(), login.state)
	s.Error(err, "the refused login spends its one-time state")
	stored := s.storedConnection(connection.ID)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(replacement.Revision, stored.Revision)
	s.Equal("replacement-account-token", s.openCredentials(stored).AccessToken)
}

func (s *ConnectorsSuite) TestTheBrowserHandoffSetsTheLoginCookieOnlyForTheRoutersOwnOrigin() {
	connection := s.seedConnection("slack", "https://mcp.slack.com/mcp")
	state := s.utils.uuid()
	login := s.startedLogin(connection, mcp.PendingAuthorization{
		State:        state,
		CodeVerifier: "pkce-verifier",
		AuthorizeURL: "https://slack.example/authorize?state=" + state,
	})
	path := connectorOAuthLaunchPath + login.attemptID

	refused, _ := s.browse(s.request(http.MethodPost, path, "https://attacker.example", `{"handoff_token":"wrong"}`))
	s.Equal(http.StatusForbidden, refused.StatusCode, "another origin cannot plant the login cookie")

	response, body := s.browse(s.request(http.MethodPost, path, s.server.URL, `{"handoff_token":"`+login.binding+`"}`))
	s.Require().Equal(http.StatusOK, response.StatusCode, string(body))
	var launched struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	s.Require().NoError(json.Unmarshal(body, &launched))
	s.Equal("https://slack.example/authorize?state="+state, launched.AuthorizationURL)
	cookies := response.Cookies()
	s.Require().Len(cookies, 1)
	s.Equal(connectorAuthorizationCookieName(login.attemptID), cookies[0].Name)
	s.Equal(login.binding, cookies[0].Value)
	s.Equal(mcp.CallbackPath, cookies[0].Path)
}

// KeylessConnectorsSuite runs the router with no key to seal connector credentials with,
// which is a deployment that never configured ROUTER_AUTH_KEK.
type KeylessConnectorsSuite struct {
	RouterSuite
}

func TestKeylessConnectorsSuite(t *testing.T) {
	runSuite(t, new(KeylessConnectorsSuite))
}

func (s *KeylessConnectorsSuite) SetupSuite() {
	s.withoutCredentialSealer = true
	s.RouterSuite.SetupSuite()
}

func (s *KeylessConnectorsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *KeylessConnectorsSuite) TestAConnectorWithoutAuthenticationCanBeActivatedWithoutAKey() {
	connection := store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: "custom_public_catalog", OwnerType: "app",
		Endpoint: "https://example.com/mcp", AuthType: connectors.AuthNone,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(context.Background(), &connection))

	status, payload := s.serverClient.call(http.MethodPut,
		"/v1/agents/connections/"+connection.ID+"/credentials", map[string]any{"expected_revision": 1})
	s.Require().Equal(http.StatusOK, status, string(payload))

	stored, err := s.store.ConnectorConnection(context.Background(), s.customerID(), connection.ID)
	s.Require().NoError(err)
	s.Equal(store.ConnectorConnected, stored.Status)
	s.Equal(connectors.AuthNone, stored.AuthType)
	s.Empty(stored.CredentialSealed)
}

func (s *KeylessConnectorsSuite) TestAConnectorThatNeedsACredentialCannotBeGivenOneWithoutAKey() {
	connection := store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: "custom_crm", OwnerType: "app",
		Endpoint: "https://example.com/mcp", AuthType: connectors.AuthBearer,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(context.Background(), &connection))

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/connections/"+connection.ID+"/credentials",
		map[string]any{"expected_revision": 1, "bearer_token": "secret-" + s.utils.uuid()})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "encrypted credential storage is unavailable")
}

// temporarilyUnavailable is a provider that cannot refresh a grant just now.
func temporarilyUnavailable(w http.ResponseWriter, _ context.CancelFunc) {
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = w.Write([]byte(`{"error":"temporarily_unavailable"}`))
}

// refresh is what using a grant that had to be refreshed came to.
type refresh struct {
	credentials connectors.Credentials
	err         error
	// stored is the connection as the refresh left it.
	stored store.ConnectorConnection
	// provider is where the grant is refreshed, and requests how often it was asked to.
	provider *httptest.Server
	requests *atomic.Int32
}

// refreshAgainst uses a grant expiring at expires whose refresh the provider answers with
// answer, which is handed the cancel of the caller's context.
func (s *ConnectorsSuite) refreshAgainst(expires time.Time, answer func(http.ResponseWriter, context.CancelFunc)) refresh {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	connection := s.seedConnection("github", "https://api.githubcopilot.com/mcp/")
	checkpointed := make(chan bool, 2)
	requests := new(atomic.Int32)
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		stored, err := s.store.ConnectorConnection(context.Background(), s.customerID(), connection.ID)
		checkpointed <- err == nil && stored.Status == store.ConnectorNeedsReauth
		answer(w, cancel)
	}))
	s.T().Cleanup(provider.Close)

	connection.Status = store.ConnectorConnected
	connection.ExpiresAt = &expires
	var err error
	connection.CredentialSealed, err = connectors.SealCredentials(s.sealer, s.customerID(), connection.ID,
		connection.Revision, connectors.Credentials{
			AuthType: connectors.AuthOAuth2, AccessToken: "old-access", RefreshToken: "old-refresh",
			OAuthClientID: "test-client", TokenEndpoint: provider.URL,
		})
	s.Require().NoError(err)
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(context.Background(), &connection, connection.Revision))

	got, err := connectors.ResolveCredentials(ctx, s.store, s.sealer, s.customerID(), connection.ID,
		&mcp.OAuthClient{HTTP: provider.Client()})
	s.True(<-checkpointed, "the refresh is checkpointed durably before the provider receives the token")
	return refresh{
		credentials: got, err: err, stored: s.storedConnection(connection.ID),
		provider: provider, requests: requests,
	}
}

// assertRefreshIsUncertain checks that a refresh whose outcome the router never learned
// leaves the grant needing reauthorization, and is not tried again: the provider may
// already have rotated the refresh token, and redeeming the old one twice is what a
// provider revokes a grant for.
func (s *ConnectorsSuite) assertRefreshIsUncertain(answer func(http.ResponseWriter, context.CancelFunc)) {
	refresh := s.refreshAgainst(time.Now().Add(30*time.Second), answer)

	s.ErrorIs(refresh.err, connectors.ErrReauthorizationRequired)
	s.Empty(refresh.credentials.AccessToken)
	s.Equal(store.ConnectorNeedsReauth, refresh.stored.Status)
	s.Contains(refresh.stored.LastError, "did not finish durably")
	_, err := connectors.ResolveCredentials(context.Background(), s.store, s.sealer, s.customerID(),
		refresh.stored.ID, &mcp.OAuthClient{HTTP: refresh.provider.Client()})
	s.ErrorIs(err, connectors.ErrReauthorizationRequired)
	s.Equal(int32(1), refresh.requests.Load(), "an uncertain rotating token is not tried again")
}

// catalog is the connectors matching a filter, or all of them for an empty one.
func (s *ConnectorsSuite) catalog(query string) []ConnectorDefinition {
	path := "/v1/agents/connectors"
	if query != "" {
		path += "?q=" + url.QueryEscape(query)
	}
	var listed []ConnectorDefinition
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, path, nil, &listed))
	return listed
}

func named(listed []ConnectorDefinition) []string {
	ids := make([]string, 0, len(listed))
	for _, connector := range listed {
		ids = append(ids, connector.Id)
	}
	return ids
}

// ownedBy is the body creating a connection to connectorID that the end user userID owns.
func (s *ConnectorsSuite) ownedBy(connectorID, userID string) map[string]any {
	return map[string]any{"connector_id": connectorID, "owner": map[string]any{"type": "user", "user_id": userID}}
}

// seedDefinition registers a connector of the app's own straight into Postgres, for an
// endpoint the API would refuse because it is not a public HTTPS host.
func (s *ConnectorsSuite) seedDefinition(id, endpoint, authType string) {
	s.Require().NoError(s.store.CreateConnectorDefinition(context.Background(), &store.ConnectorDefinition{
		CustomerID: s.customerID(), ID: id, Name: id, Endpoint: endpoint, AuthType: authType,
	}))
}

// seedConnection is an app-owned connection to connectorID waiting to be connected, made
// straight in Postgres so that nothing resolves the endpoint's host.
func (s *ConnectorsSuite) seedConnection(connectorID, endpoint string) store.ConnectorConnection {
	connection := store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: connectorID, OwnerType: "app",
		Endpoint: endpoint, AuthType: connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(context.Background(), &connection))
	return s.storedConnection(connection.ID)
}

// seedOwnedConnection is a Gong connection the end user ownerID owns.
func (s *ConnectorsSuite) seedOwnedConnection(ownerID string) store.ConnectorConnection {
	connection := store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: "gong", OwnerType: "user", OwnerID: ownerID,
		Endpoint: "https://mcp.gong.io/mcp", AuthType: connectors.AuthOAuth2,
	}
	s.Require().NoError(s.store.CreateConnectorConnection(context.Background(), &connection))
	return s.storedConnection(connection.ID)
}

// connectWith gives a connection a grant sealed with sealer at its next revision, the way a
// finished login leaves one, and returns it as stored.
func (s *ConnectorsSuite) connectWith(
	sealer *auth.Sealer, connection store.ConnectorConnection, credentials connectors.Credentials,
) store.ConnectorConnection {
	previous := connection.Revision
	connection.Revision++
	connection.Status = store.ConnectorConnected
	connection.CredentialKEKVersion = sealer.CurrentVersion()
	sealed, err := connectors.SealCredentials(sealer, s.customerID(), connection.ID, connection.Revision, credentials)
	s.Require().NoError(err)
	connection.CredentialSealed = sealed
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(context.Background(), &connection, previous))
	return s.storedConnection(connection.ID)
}

func (s *ConnectorsSuite) storedConnection(id string) store.ConnectorConnection {
	stored, err := s.store.ConnectorConnection(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	return stored
}

// openCredentials unseals what a connection holds, which only the router's key can.
func (s *ConnectorsSuite) openCredentials(connection store.ConnectorConnection) connectors.Credentials {
	credentials, err := connectors.OpenCredentials(s.sealer, s.customerID(), connection.ID, connection.Revision,
		connection.CredentialKEKVersion, connection.CredentialSealed)
	s.Require().NoError(err)
	return credentials
}

// login is one somebody started, as authorizing a connection leaves it: a one-time state,
// and the secret only the browser that started it holds.
type login struct {
	attemptID, binding, state string
}

// cookie is what the browser that started the login carries back from the provider.
func (l login) cookie() *http.Cookie {
	return &http.Cookie{Name: connectorAuthorizationCookieName(l.attemptID), Value: l.binding}
}

// startedLogin writes down a login for connection at its current revision. The pending
// authorization's state is a fresh one unless the test names its own.
func (s *ConnectorsSuite) startedLogin(connection store.ConnectorConnection, pending mcp.PendingAuthorization) login {
	if pending.State == "" {
		pending.State = s.utils.uuid()
	}
	started := login{attemptID: s.utils.uuid(), binding: s.utils.uuid(), state: pending.State}
	sealed, err := connectors.SealAuthorizationAttempt(s.sealer, started.attemptID, connectors.AuthorizationAttempt{
		ConnectionID:   connection.ID,
		Revision:       connection.Revision,
		ConnectorID:    connection.ConnectorID,
		BrowserBinding: started.binding,
		Pending:        pending,
	})
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(context.Background(), &store.ConnectorAuthorizationAttempt{
		ID:            started.attemptID,
		CustomerID:    s.customerID(),
		ConnectionID:  connection.ID,
		StateHash:     store.OAuthStateHash(pending.State),
		AttemptSealed: sealed,
		KEKVersion:    s.sealer.CurrentVersion(),
		ExpiresAt:     time.Now().Add(time.Minute),
	}))
	return started
}

// arrive is the browser coming back from the provider with query, carrying cookie when it
// has one.
func (s *ConnectorsSuite) arrive(query url.Values, cookie *http.Cookie) *http.Response {
	request := s.request(http.MethodGet, mcp.CallbackPath+"?"+query.Encode(), "", "")
	if cookie != nil {
		request.AddCookie(cookie)
	}
	response, _ := s.browse(request)
	return response
}

// dashboardReturn is the query of where a finished login sent the browser, which has to be
// the configured dashboard page.
func (s *ConnectorsSuite) dashboardReturn(response *http.Response) url.Values {
	location, err := url.Parse(response.Header.Get("Location"))
	s.Require().NoError(err)
	s.Equal("https://dashboard.test/connections", location.Scheme+"://"+location.Host+location.Path)
	s.Equal("agent", location.Query().Get("config_id"), "the dashboard's own query is kept")
	return location.Query()
}

// request is one a browser makes, holding none of the app's credentials: from origin when it
// names one, with body as JSON when there is one.
func (s *ConnectorsSuite) request(method, path, origin, body string) *http.Request {
	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}
	request, err := http.NewRequest(method, s.server.URL+path, reader)
	s.Require().NoError(err)
	if origin != "" {
		request.Header.Set("Origin", origin)
	}
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	return request
}

// browse sends a browser's request without following where it is redirected, since where
// the browser is sent is what a test reads.
func (s *ConnectorsSuite) browse(request *http.Request) (*http.Response, []byte) {
	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := browser.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response, body
}
