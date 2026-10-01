package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/suite"
)

type OAuthSuite struct {
	suite.Suite
}

func TestOAuthSuite(t *testing.T) {
	suite.Run(t, new(OAuthSuite))
}

func (s *OAuthSuite) TestConfidentialOAuthRequiresBothClientCredentials() {
	client := &OAuthClient{}
	connector := Connector{ID: "slack", URL: "https://mcp.slack.com/mcp", OAuthMode: "confidential"}
	for _, credentials := range [][2]string{{"", ""}, {"client-id", ""}, {"", "client-secret"}} {
		_, err := client.StartAuthorizeWithClient(context.Background(), connector, "", credentials[0], credentials[1])
		s.ErrorIs(err, ErrOAuthClientNotConfigured)
	}
}

func (s *OAuthSuite) TestSalesforceOAuthEndpointsFollowTheSelectedInstance() {
	production := authServer{
		AuthorizationEndpoint: "https://login.salesforce.com/services/oauth2/authorize",
		TokenEndpoint:         "https://login.salesforce.com/services/oauth2/token",
	}
	connector := Connector{ID: "salesforce"}

	s.Equal(production, providerOAuthEndpoints(connector, "production", production))
	s.Equal(production, providerOAuthEndpoints(connector, "", production))
	sandbox := providerOAuthEndpoints(connector, "sandbox", production)
	s.Equal("https://test.salesforce.com/services/oauth2/authorize", sandbox.AuthorizationEndpoint)
	s.Equal("https://test.salesforce.com/services/oauth2/token", sandbox.TokenEndpoint)
	s.Equal(sandbox, providerOAuthEndpoints(connector, "TEST", production))

	otherProvider := providerOAuthEndpoints(Connector{ID: "github"}, "sandbox", production)
	s.Equal(production, otherProvider)
}

func (s *OAuthSuite) TestOAuthResponsesAreBoundedAndRegistrationErrorsAreRedacted() {
	mux := http.NewServeMux()
	mux.HandleFunc("/oversized", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(strings.Repeat("x", maxOAuthResponseBytes+1)))
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte("private-client-secret"))
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	var metadata map[string]any
	err := getJSON(context.Background(), server.Client(), server.URL+"/oversized", &metadata)
	s.ErrorContains(err, "OAuth response exceeded size limit")

	auth := &OAuthClient{HTTP: server.Client()}
	_, err = auth.register(context.Background(), server.Client(), server.URL+"/register", "none")
	s.ErrorContains(err, "HTTP 502")
	s.NotContains(err.Error(), "private-client-secret")
}

func (s *OAuthSuite) TestTokenEndpointsCannotAcceptCredentialsFromAnErrorStatus() {
	mux := http.NewServeMux()
	mux.HandleFunc("/token", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(tokenResponse{AccessToken: "unexpected-token"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client()}
	_, exchangeErr := auth.Exchange(context.Background(), PendingAuthorization{
		ClientID: "client", TokenEndpoint: server.URL + "/token",
	}, "code")
	s.ErrorContains(exchangeErr, "HTTP 502")

	_, refreshErr := auth.RefreshWithClient(context.Background(), PendingAuthorization{
		ClientID: "client", TokenEndpoint: server.URL + "/token",
	}, "refresh-token")
	s.ErrorContains(refreshErr, "HTTP 502")
}

func (s *OAuthSuite) TestStartAuthorizeUsesDiscoveryAndDCR() {
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                   server.URL,
			CodeChallengeMethods:     []string{"S256"},
			AuthorizationEndpoint:    server.URL + "/authorize",
			TokenEndpoint:            server.URL + "/token",
			RegistrationEndpoint:     server.URL + "/register",
			TokenEndpointAuthMethods: []string{"client_secret_basic"},
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		s.Equal(http.MethodPost, r.Method)
		var body map[string]any
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		s.Equal("client_secret_basic", body["token_endpoint_auth_method"])
		_ = json.NewEncoder(w).Encode(registration{
			ClientID: "dyn-1", ClientSecret: "secret-1", TokenEndpointAuthMethod: "client_secret_basic",
		})
	})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		username, password, ok := r.BasicAuth()
		s.True(ok)
		s.Equal("dyn-1", username)
		s.Equal("secret-1", password)
		_ = json.NewEncoder(w).Encode(tokenResponse{AccessToken: "access-1"})
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{
		HTTP:      server.Client(),
		PublicURL: "http://router.example",
	}
	// Point Slack's origin discovery at the test server by using a connector whose
	// endpoint is the test server itself.
	connector := Connector{ID: "slack", Name: "Slack", URL: server.URL + "/mcp"}
	pending, err := auth.StartAuthorize(context.Background(), connector, "")
	s.Require().NoError(err)
	s.Equal("dyn-1", pending.ClientID)
	s.Contains(pending.AuthorizeURL, "client_id=dyn-1")
	s.Contains(pending.AuthorizeURL, "code_challenge")
	s.Equal(server.URL+"/token", pending.TokenEndpoint)
	s.Equal("client_secret_basic", pending.ClientAuthMethod)

	_, err = auth.Exchange(context.Background(), pending, "code-1")
	s.Require().NoError(err)
}

func (s *OAuthSuite) TestStartAuthorizePrefersCIMDAndDiscoversOIDCMetadata() {
	var server *httptest.Server
	registerCalls := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                            server.URL,
			AuthorizationEndpoint:             server.URL + "/authorize",
			TokenEndpoint:                     server.URL + "/token",
			JWKSURI:                           server.URL + "/jwks",
			ResponseTypesSupported:            []string{"code"},
			RegistrationEndpoint:              server.URL + "/register",
			TokenEndpointAuthMethods:          []string{"none"},
			CodeChallengeMethods:              []string{"S256"},
			ClientIDMetadataSupported:         true,
			AuthorizationResponseIssSupported: true,
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, _ *http.Request) {
		registerCalls++
		w.WriteHeader(http.StatusInternalServerError)
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{ID: "custom_demo", Name: "Demo", URL: server.URL + "/mcp", OAuthMode: "dcr"}
	pending, err := auth.StartAuthorize(context.Background(), connector, "")
	s.Require().NoError(err)
	s.Equal(server.URL, pending.Issuer)
	s.True(pending.RequireIssuer)
	s.Equal("https://router.example"+ClientMetadataPath, pending.ClientID)
	s.Equal("none", pending.ClientAuthMethod)
	s.Equal(0, registerCalls)

	authorizeURL, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal(pending.ClientID, authorizeURL.Query().Get("client_id"))
	s.Equal("S256", authorizeURL.Query().Get("code_challenge_method"))

	document := auth.ClientMetadataDocument()
	s.Equal(pending.ClientID, document.ClientID)
	s.Equal([]string{"https://router.example" + CallbackPath}, document.RedirectURIs)
	s.Equal("none", document.TokenEndpointAuthMethod)

	s.Error(pending.ValidateAuthorizationResponseIssuer(""))
	s.Error(pending.ValidateAuthorizationResponseIssuer("https://attacker.example"))
	s.NoError(pending.ValidateAuthorizationResponseIssuer(server.URL))
}

func (s *OAuthSuite) TestStartAuthorizeRequiresPKCES256() {
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                server.URL,
			AuthorizationEndpoint: server.URL + "/authorize",
			TokenEndpoint:         server.URL + "/token",
			CodeChallengeMethods:  []string{"plain"},
		})
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{ID: "custom_demo", Name: "Demo", URL: server.URL + "/mcp", OAuthMode: "dcr"}
	_, err := auth.StartAuthorize(context.Background(), connector, "")
	s.ErrorContains(err, "does not support OAuth PKCE with S256")
}

func (s *OAuthSuite) TestDCRRefusesAnUnsupportedTokenAuthenticationMethod() {
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/.well-known/oauth-authorization-server" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(authServer{
				Issuer:                   server.URL,
				CodeChallengeMethods:     []string{"S256"},
				AuthorizationEndpoint:    server.URL + "/authorize",
				TokenEndpoint:            server.URL + "/token",
				RegistrationEndpoint:     server.URL + "/register",
				TokenEndpointAuthMethods: []string{"private_key_jwt"},
			})
		}
	}))
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{ID: "custom_demo", Name: "Demo", URL: server.URL + "/mcp", OAuthMode: "dcr"}
	_, err := auth.StartAuthorize(context.Background(), connector, "")
	s.ErrorContains(err, "no supported OAuth client authentication method")
}

func (s *OAuthSuite) TestExchangeStoresTheAccessToken() {
	mux := http.NewServeMux()
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		s.Equal("authorization_code", r.FormValue("grant_type"))
		s.Equal("abc", r.FormValue("code"))
		_ = json.NewEncoder(w).Encode(tokenResponse{
			AccessToken:  "tok-1",
			RefreshToken: "ref-1",
			ExpiresIn:    3600,
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "http://router.example"}
	token, err := auth.Exchange(context.Background(), PendingAuthorization{
		ClientID:      "dyn-1",
		CodeVerifier:  "ver",
		TokenEndpoint: server.URL + "/token",
	}, "abc")
	s.Require().NoError(err)
	s.Equal("tok-1", token.AccessToken)
	s.Equal("ref-1", token.RefreshToken)
	s.NotNil(token.ExpiresAt)
}

func (s *OAuthSuite) TestOAuthTokenResponsesProvideStableAccountIDsWhenAvailable() {
	tests := []struct {
		name          string
		connectorID   string
		tokenEndpoint string
		body          tokenResponse
		want          string
	}{
		{
			name:          "Slack workspace and member",
			connectorID:   "slack",
			tokenEndpoint: "https://slack.com/api/oauth.v2.user.access",
			body: tokenResponse{
				Team: struct {
					ID string `json:"id"`
				}{ID: "T123"},
				AuthedUser: struct {
					ID           string `json:"id"`
					AccessToken  string `json:"access_token"`
					RefreshToken string `json:"refresh_token"`
					ExpiresIn    int    `json:"expires_in"`
					Scope        string `json:"scope"`
				}{ID: "U456"},
			},
			want: "slack:T123:U456",
		},
		{
			name:          "Calendly user and organization",
			connectorID:   "calendly",
			tokenEndpoint: "https://auth.calendly.com/oauth/token",
			body: tokenResponse{
				Owner:        "https://api.calendly.com/users/U123",
				Organization: "https://api.calendly.com/organizations/O456",
			},
			want: "calendly:U123:O456",
		},
		{
			name:          "Salesforce organization and user",
			connectorID:   "salesforce",
			tokenEndpoint: "https://login.salesforce.com/services/oauth2/token",
			body:          tokenResponse{ID: "https://login.salesforce.com/id/ORG123/USER456"},
			want:          "salesforce:ORG123:USER456",
		},
		{
			name:          "Salesforce identity URL on another host",
			connectorID:   "salesforce",
			tokenEndpoint: "https://login.salesforce.com/services/oauth2/token",
			body:          tokenResponse{ID: "https://attacker.example/id/ORG123/USER456"},
		},
		{
			name:          "Calendly owner URL on another host",
			connectorID:   "calendly",
			tokenEndpoint: "https://auth.calendly.com/oauth/token",
			body:          tokenResponse{Owner: "https://attacker.example/users/U123"},
		},
	}
	for _, test := range tests {
		s.Run(test.name, func() {
			s.Equal(test.want, providerAccountID(test.connectorID, test.tokenEndpoint, test.body))
		})
	}
}

func (s *OAuthSuite) TestConfidentialOAuthRequestsScopedSlackConsentAndBindsTheResource() {
	var exchange url.Values
	mux := http.NewServeMux()
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		s.Require().NoError(r.ParseForm())
		exchange = r.Form
		_ = json.NewEncoder(w).Encode(tokenResponse{
			AccessToken:  "tok-2",
			RefreshToken: "ref-2",
			Scope:        "search:read.public,chat:write",
			Team: struct {
				ID string `json:"id"`
			}{ID: "T123"},
			AuthedUser: struct {
				ID           string `json:"id"`
				AccessToken  string `json:"access_token"`
				RefreshToken string `json:"refresh_token"`
				ExpiresIn    int    `json:"expires_in"`
				Scope        string `json:"scope"`
			}{ID: "U456"},
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{
		ID:                    "slack",
		Name:                  "Slack",
		URL:                   server.URL + "/mcp",
		OAuthMode:             "confidential",
		Resource:              server.URL,
		AuthorizationEndpoint: server.URL + "/authorize",
		TokenEndpoint:         server.URL + "/token",
		Scopes:                []string{"search:read.public", "chat:write"},
	}
	pending, err := auth.StartAuthorizeWithClient(context.Background(), connector, "", "client", "secret")
	s.Require().NoError(err)
	parsedAuthorizeURL, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal("https://router.example"+CallbackPath, parsedAuthorizeURL.Query().Get("redirect_uri"))
	s.Equal("search:read.public,chat:write", parsedAuthorizeURL.Query().Get("scope"))
	s.Empty(parsedAuthorizeURL.Query().Get("user_scope"))
	s.Equal(server.URL, parsedAuthorizeURL.Query().Get("resource"))

	token, err := auth.Exchange(context.Background(), pending, "auth-code")
	s.Require().NoError(err)
	s.Equal("secret", exchange.Get("client_secret"))
	s.Equal(server.URL, pending.Resource)
	s.Equal(pending.Resource, exchange.Get("resource"))
	s.Equal([]string{"search:read.public", "chat:write"}, token.Scopes)
	s.Equal("tok-2", token.AccessToken)
	s.Equal("slack:T123:U456", token.AccountID)
}

func (s *OAuthSuite) TestCustomerConfidentialOAuthUsesConfiguredBasicAuthentication() {
	var exchange url.Values
	var grantTypes []string
	mux := http.NewServeMux()
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		username, password, ok := r.BasicAuth()
		s.True(ok)
		s.Equal("gong-client", username)
		s.Equal("gong-secret", password)
		s.Require().NoError(r.ParseForm())
		exchange = r.Form
		grantTypes = append(grantTypes, exchange.Get("grant_type"))
		_ = json.NewEncoder(w).Encode(tokenResponse{
			AccessToken:  "gong-access",
			RefreshToken: "gong-refresh",
			ExpiresIn:    3600,
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{
		ID:                      "gong",
		Name:                    "Gong",
		URL:                     server.URL + "/mcp",
		OAuthMode:               "customer_dcr",
		AuthorizationEndpoint:   server.URL + "/authorize",
		TokenEndpoint:           server.URL + "/token",
		RegistrationEndpoint:    server.URL + "/register",
		TokenEndpointAuthMethod: "client_secret_basic",
	}
	pending, err := auth.StartAuthorizeWithClient(context.Background(), connector, "", "gong-client", "gong-secret")
	s.Require().NoError(err)
	s.Equal("client_secret_basic", pending.ClientAuthMethod)

	token, err := auth.Exchange(context.Background(), pending, "auth-code")
	s.Require().NoError(err)
	s.Empty(exchange.Get("client_id"))
	s.Empty(exchange.Get("client_secret"))
	s.Equal("authorization_code", exchange.Get("grant_type"))
	s.Equal("gong-access", token.AccessToken)
	s.Equal("gong-refresh", token.RefreshToken)

	refreshed, err := auth.RefreshWithClient(context.Background(), pending, token.RefreshToken)
	s.Require().NoError(err)
	s.Equal([]string{"authorization_code", "refresh_token"}, grantTypes)
	s.Equal("gong-access", refreshed.AccessToken)
}

func (s *OAuthSuite) TestCustomerDCRSupportsAutomaticGongRegistrationWithoutClientCredentials() {
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{AuthorizationServers: []string{server.URL}})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                   server.URL,
			CodeChallengeMethods:     []string{"S256"},
			AuthorizationEndpoint:    server.URL + "/authorize",
			TokenEndpoint:            server.URL + "/token",
			RegistrationEndpoint:     server.URL + "/register",
			TokenEndpointAuthMethods: []string{"none"},
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&body))
		s.Equal("none", body["token_endpoint_auth_method"])
		_ = json.NewEncoder(w).Encode(registration{ClientID: "auto-client", TokenEndpointAuthMethod: "none"})
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{
		ID:                    "gong",
		Name:                  "Gong",
		URL:                   server.URL + "/mcp",
		OAuthMode:             "customer_dcr",
		AuthorizationEndpoint: server.URL + "/authorize",
		TokenEndpoint:         server.URL + "/token",
	}
	pending, err := auth.StartAuthorize(context.Background(), connector, "")
	s.Require().NoError(err)
	s.Equal("auto-client", pending.ClientID)
	s.Equal("none", pending.ClientAuthMethod)
}

func (s *OAuthSuite) TestDCRCanUseAnOperatorConfiguredPublicOAuthClientID() {
	s.T().Setenv("GITHUB_MCP_CLIENT_ID", "github-public-client")
	s.T().Setenv("GITHUB_MCP_CLIENT_SECRET", "")
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{
			Resource:             server.URL + "/mcp",
			AuthorizationServers: []string{server.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                   server.URL,
			CodeChallengeMethods:     []string{"S256"},
			AuthorizationEndpoint:    server.URL + "/authorize",
			TokenEndpoint:            server.URL + "/token",
			TokenEndpointAuthMethods: []string{"none"},
		})
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client(), PublicURL: "https://router.example"}
	connector := Connector{
		ID:        "github",
		Name:      "GitHub",
		URL:       server.URL + "/mcp",
		OAuthMode: "dcr",
		ClientEnv: "GITHUB",
	}
	pending, err := auth.StartAuthorize(context.Background(), connector, "")
	s.Require().NoError(err)
	s.Equal("github-public-client", pending.ClientID)
	s.Equal("none", pending.ClientAuthMethod)
}

func (s *OAuthSuite) TestRefreshUsesTheConfidentialClientResourceAndProviderEndpoint() {
	var refresh url.Values
	mux := http.NewServeMux()
	mux.HandleFunc("/refresh", func(w http.ResponseWriter, r *http.Request) {
		s.Require().NoError(r.ParseForm())
		refresh = r.Form
		_ = json.NewEncoder(w).Encode(tokenResponse{AccessToken: "tok-3", ExpiresIn: 1800})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client()}
	token, err := auth.RefreshWithClient(context.Background(), PendingAuthorization{
		ClientID:         "client",
		ClientSecret:     "secret",
		ClientAuthMethod: "client_secret_post",
		TokenEndpoint:    server.URL + "/wrong-token-endpoint",
		RefreshEndpoint:  server.URL + "/refresh",
		Resource:         server.URL + "/mcp",
	}, "refresh-token")
	s.Require().NoError(err)
	s.Equal("secret", refresh.Get("client_secret"))
	s.Equal("refresh-token", refresh.Get("refresh_token"))
	s.Equal(server.URL+"/mcp", refresh.Get("resource"))
	s.Equal("tok-3", token.AccessToken)
	s.Equal("refresh-token", token.RefreshToken)
}

func (s *OAuthSuite) TestOAuthImportSettingsUseTheCatalogClientAndEndpoints() {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client()}
	connector := Connector{
		ID:                      "gong",
		Name:                    "Gong",
		URL:                     server.URL + "/mcp",
		OAuthMode:               "customer_dcr",
		AuthorizationEndpoint:   server.URL + "/authorize",
		TokenEndpoint:           server.URL + "/token",
		TokenEndpointAuthMethod: "client_secret_basic",
	}
	pending, err := auth.OAuthClientForImport(context.Background(), connector, "", "gong-client", "gong-secret")
	s.Require().NoError(err)
	s.Equal("gong-client", pending.ClientID)
	s.Equal("gong-secret", pending.ClientSecret)
	s.Equal("client_secret_basic", pending.ClientAuthMethod)
	s.Equal(server.URL+"/token", pending.TokenEndpoint)
	s.Equal(server.URL+"/mcp", pending.Resource)
}

func (s *OAuthSuite) TestOAuthImportSettingsUseDiscoveredResourceForAPublicClient() {
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{
			Resource:             server.URL + "/resource",
			AuthorizationServers: []string{server.URL},
		})
	})
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(authServer{
			Issuer:                   server.URL,
			CodeChallengeMethods:     []string{"S256"},
			AuthorizationEndpoint:    server.URL + "/authorize",
			TokenEndpoint:            server.URL + "/token",
			TokenEndpointAuthMethods: []string{"none"},
		})
	})
	server = httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client()}
	connector := Connector{ID: "custom_crm", Name: "CRM", URL: server.URL + "/mcp", OAuthMode: "dcr"}
	pending, err := auth.OAuthClientForImport(context.Background(), connector, "", "public-client", "")
	s.Require().NoError(err)
	s.Equal("public-client", pending.ClientID)
	s.Equal("none", pending.ClientAuthMethod)
	s.Equal(server.URL+"/resource", pending.Resource)
	s.Equal(server.URL+"/token", pending.TokenEndpoint)
}

func (s *OAuthSuite) TestRefreshDistinguishesRevocationFromTemporaryFailure() {
	mux := http.NewServeMux()
	mux.HandleFunc("/invalid", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(tokenResponse{Error: "invalid_grant", ErrorDesc: "private provider detail"})
	})
	mux.HandleFunc("/unavailable", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(tokenResponse{Error: "temporarily_unavailable", ErrorDesc: "private provider detail"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &OAuthClient{HTTP: server.Client()}
	_, invalidErr := auth.RefreshWithClient(context.Background(), PendingAuthorization{
		ClientID: "client", TokenEndpoint: server.URL + "/invalid",
	}, "refresh-token")
	s.ErrorIs(invalidErr, ErrOAuthInvalidGrant)
	s.NotContains(invalidErr.Error(), "private provider detail")

	_, unavailableErr := auth.RefreshWithClient(context.Background(), PendingAuthorization{
		ClientID: "client", TokenEndpoint: server.URL + "/unavailable",
	}, "refresh-token")
	s.Error(unavailableErr)
	s.False(errors.Is(unavailableErr, ErrOAuthInvalidGrant))
	s.False(errors.Is(unavailableErr, ErrOAuthRefreshUncertain))
	s.NotContains(unavailableErr.Error(), "private provider detail")
}

func (s *OAuthSuite) TestRefreshRecognizesInvalidSlackTokensAndUncertainResponses() {
	for _, body := range []string{
		`{"error":"invalid_refresh_token"}`,
		`{"error":"internal_error"}`,
		`{"error":"fatal_error"}`,
		`{"access_token":`,
		`{}`,
	} {
		s.Run(body, func() {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(body))
			}))
			defer server.Close()
			auth := &OAuthClient{HTTP: server.Client()}
			_, err := auth.RefreshWithClient(context.Background(), PendingAuthorization{
				ClientID: "client", TokenEndpoint: server.URL,
			}, "refresh-token")
			if strings.Contains(body, "invalid_refresh_token") {
				s.ErrorIs(err, ErrOAuthInvalidGrant)
			} else {
				s.ErrorIs(err, ErrOAuthRefreshUncertain)
			}
		})
	}
}

func (s *OAuthSuite) TestTokenExchangeDoesNotFollowCrossHostRedirects() {
	var receivedRequest atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedRequest.Store(true)
		_ = json.NewEncoder(w).Encode(tokenResponse{AccessToken: "should-not-arrive"})
	}))
	defer target.Close()
	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer source.Close()

	auth := &OAuthClient{HTTP: source.Client(), PublicURL: "https://router.example"}
	_, err := auth.Exchange(context.Background(), PendingAuthorization{
		ClientID: "client", ClientSecret: "secret", ClientAuthMethod: "client_secret_post",
		CodeVerifier: "verifier", TokenEndpoint: source.URL,
	}, "authorization-code")

	s.Error(err)
	s.False(receivedRequest.Load())
}
