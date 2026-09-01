package plugins

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
