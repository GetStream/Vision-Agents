package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/suite"
)

type OAuthSuite struct {
	suite.Suite
}

func TestOAuthSuite(t *testing.T) {
	suite.Run(t, new(OAuthSuite))
}

func (s *OAuthSuite) TestStartAuthorizeUsesDiscoveryAndDCR() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "http://auth.example/authorize",
			TokenEndpoint:         "http://auth.example/token",
			RegistrationEndpoint:  "http://" + r.Host + "/register",
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		s.Equal(http.MethodPost, r.Method)
		_ = json.NewEncoder(w).Encode(registration{ClientID: "dyn-1"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &Auth{
		HTTP:         server.Client(),
		PublicURL:    "http://router.example",
		DashboardURL: "http://dash.example",
	}
	// Point Slack's origin discovery at the test server by using a plugin whose
	// endpoint is the test server itself.
	plugin := Plugin{ID: "slack", Name: "Slack", URL: server.URL + "/mcp"}
	pending, err := auth.StartAuthorize(context.Background(), plugin, "")
	s.Require().NoError(err)
	s.Equal("dyn-1", pending.ClientID)
	s.Contains(pending.AuthorizeURL, "client_id=dyn-1")
	s.Contains(pending.AuthorizeURL, "code_challenge")
	s.Equal("http://auth.example/token", pending.TokenEndpoint)
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

	auth := &Auth{HTTP: server.Client(), PublicURL: "http://router.example"}
	token, err := auth.Exchange(context.Background(), Pending{
		ClientID:      "dyn-1",
		CodeVerifier:  "ver",
		TokenEndpoint: server.URL + "/token",
	}, "abc")
	s.Require().NoError(err)
	s.Equal("tok-1", token.AccessToken)
	s.Equal("ref-1", token.RefreshToken)
	s.NotNil(token.ExpiresAt)
}

func (s *OAuthSuite) TestMetadataAtTheEndpointsPathIsFoundFirst() {
	// Where Sentry and Google Calendar publish it (RFC 9728 section 3.1): their origin's
	// well-known URL is a 404.
	mux := http.NewServeMux()
	var issuer string
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp/v1", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{AuthorizationServers: []string{issuer + "/issuer"}})
	})
	mux.HandleFunc("/issuer/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://accounts.example/auth",
			TokenEndpoint:         "https://accounts.example/token",
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	issuer = server.URL
	s.T().Setenv("CALENDAR_MCP_CLIENT_ID", "web-client")

	auth := &Auth{HTTP: server.Client(), PublicURL: "https://router.example"}
	pending, err := auth.StartAuthorize(context.Background(), Plugin{
		ID: "calendar", Name: "Calendar", URL: server.URL + "/mcp/v1",
		Scopes:          []string{"calendar.readonly", "calendar.events.readonly"},
		AuthorizeParams: map[string]string{"access_type": "offline"},
	}, "")
	s.Require().NoError(err)

	authorize, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal("accounts.example", authorize.Host)
	s.Equal("web-client", authorize.Query().Get("client_id"))
	s.Equal("calendar.readonly calendar.events.readonly", authorize.Query().Get("scope"))
	s.Equal("offline", authorize.Query().Get("access_type"))
	s.Equal("calendar", pending.PluginID)
}

func (s *OAuthSuite) TestTheDeploymentsOwnClientSendsItsSecret() {
	s.T().Setenv("CALENDAR_MCP_CLIENT_ID", "web-client")
	s.T().Setenv("CALENDAR_MCP_CLIENT_SECRET", "web-secret")
	var secrets []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		secrets = append(secrets, r.FormValue("client_secret"))
		_ = json.NewEncoder(w).Encode(tokenResponse{AccessToken: "tok", RefreshToken: "ref"})
	}))
	defer server.Close()
	auth := &Auth{HTTP: server.Client()}

	_, err := auth.Exchange(context.Background(), Pending{
		PluginID: "calendar", ClientID: "web-client", TokenEndpoint: server.URL,
	}, "code")
	s.Require().NoError(err)
	_, err = auth.Refresh(context.Background(), "calendar", server.URL, "web-client", "ref")
	s.Require().NoError(err)
	_, err = auth.Exchange(context.Background(), Pending{
		PluginID: "calendar", ClientID: "registered-on-the-fly", TokenEndpoint: server.URL,
	}, "code")
	s.Require().NoError(err)

	s.Equal([]string{"web-secret", "web-secret", ""}, secrets,
		"a client registered on the fly is not the one the secret belongs to")
}
