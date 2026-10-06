package plugins

import (
	"context"
	"encoding/json"
	"errors"
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

func (s *OAuthSuite) TestDashboardRedirectNamesTheConnectedPlugin() {
	auth := &Auth{DashboardURL: "https://dash.example/org/1/app/2/agents/"}
	s.Equal("https://dash.example/org/1/app/2/agents/agents/cfg-1?plugin_connected=sentry",
		auth.DashboardRedirect("cfg-1", "sentry"))
}

// GitHub publishes no RFC 8414 metadata for github.com/login/oauth, only OpenID Connect
// discovery, so a provider that has one and not the other has to be followed too.
func (s *OAuthSuite) TestStartAuthorizeFallsBackToOpenIDDiscovery() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		http.NotFound(w, nil)
	})
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://github.com/login/oauth/authorize",
			TokenEndpoint:         "https://github.com/login/oauth/access_token",
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &Auth{HTTP: server.Client(), PublicURL: "http://router.example"}
	plugin := Plugin{ID: "github", Name: "GitHub", URL: server.URL + "/mcp", Scopes: []string{"repo"}}
	// No registration endpoint and no env client, so the login says what is missing
	// rather than reaching GitHub's authorize page without a client.
	_, err := auth.StartAuthorize(context.Background(), plugin, "")
	s.ErrorContains(err, "GITHUB_MCP_CLIENT_ID")

	s.T().Setenv("GITHUB_MCP_CLIENT_ID", "gh-app")
	pending, err := auth.StartAuthorize(context.Background(), plugin, "")
	s.Require().NoError(err)
	s.Equal("https://github.com/login/oauth/access_token", pending.TokenEndpoint)
	s.Contains(pending.AuthorizeURL, "https://github.com/login/oauth/authorize?")
	s.Contains(pending.AuthorizeURL, "client_id=gh-app")
	s.Contains(pending.AuthorizeURL, "scope=repo")
}

func (s *OAuthSuite) TestADiscoveryMissWithNeitherDocumentIsReported() {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()

	auth := &Auth{HTTP: server.Client(), PublicURL: "http://router.example"}
	_, err := auth.StartAuthorize(context.Background(),
		Plugin{ID: "carrier-pigeon", Name: "Pigeon", URL: server.URL + "/mcp"}, "")

	s.ErrorContains(err, "oauth discovery")
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

// The read-only endpoint is its own resource, so the token has to be minted for it rather
// than for the endpoint that writes.
func (s *OAuthSuite) TestAReadonlyLoginIsForTheReadOnlyResourceAndAsksOnlyToRead() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://auth.example/authorize",
			TokenEndpoint:         "https://auth.example/token",
			RegistrationEndpoint:  "http://" + r.Host + "/register",
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(registration{ClientID: "dyn-1"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	plugin, err := Plugin{
		ID: "tracker", Name: "Tracker", URL: server.URL + "/mcp", Scopes: []string{"read", "write"},
		ReadonlyURL: server.URL + "/mcp/readonly", ReadonlyScopes: []string{"read"},
	}.Configured(Options{Readonly: true})
	s.Require().NoError(err)

	auth := &Auth{HTTP: server.Client(), PublicURL: "https://router.example"}
	pending, err := auth.StartAuthorize(context.Background(), plugin, "")
	s.Require().NoError(err)

	authorize, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal(server.URL+"/mcp/readonly", authorize.Query().Get("resource"))
	s.Equal("read", authorize.Query().Get("scope"))
}

// The toolsets are a choice of tools on the server, not another server, so the login is
// for the server and survives a change of toolsets.
func (s *OAuthSuite) TestALoginForSomeToolsetsIsForTheWholeServer() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://auth.example/authorize",
			TokenEndpoint:         "https://auth.example/token",
			RegistrationEndpoint:  "http://" + r.Host + "/register",
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(registration{ClientID: "dyn-1"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	plugin, err := Plugin{
		ID: "scheduler", Name: "Scheduler", URL: server.URL + "/mcp", Toolsets: []string{"bookings"},
	}.Configured(Options{Toolsets: []string{"bookings"}})
	s.Require().NoError(err)

	auth := &Auth{HTTP: server.Client(), PublicURL: "https://router.example"}
	pending, err := auth.StartAuthorize(context.Background(), plugin, "")
	s.Require().NoError(err)

	authorize, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal(server.URL+"/mcp", authorize.Query().Get("resource"))
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

// A server named by URL that publishes its metadata at neither well-known path still says
// where it is when it refuses a request without a login (RFC 9728 section 5.1), and a login
// given no scopes asks for the ones it advertises there.
func (s *OAuthSuite) TestAServerNamedByURLIsFoundThroughWhatItSaysWhenItRefusesALogin() {
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="http://`+r.Host+`/meta/notes"`)
		http.Error(w, "log in", http.StatusUnauthorized)
	})
	mux.HandleFunc("/meta/notes", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{
			AuthorizationServers: []string{"http://" + r.Host + "/issuer"},
			ScopesSupported:      []string{"notes.read", "notes.write"},
		})
	})
	mux.HandleFunc("/issuer/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://notes.example/authorize",
			TokenEndpoint:         "https://notes.example/token",
			RegistrationEndpoint:  "http://" + r.Host + "/register",
		})
	})
	mux.HandleFunc("/register", func(w http.ResponseWriter, r *http.Request) {
		var asked map[string]any
		s.Require().NoError(json.NewDecoder(r.Body).Decode(&asked))
		s.Equal([]any{"https://router.example" + CallbackPath}, asked["redirect_uris"])
		_ = json.NewEncoder(w).Encode(registration{ClientID: "notes-client"})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &Auth{HTTP: server.Client(), PublicURL: "https://router.example"}
	pending, err := auth.StartAuthorize(context.Background(),
		Plugin{ID: "notes", Name: "notes", URL: server.URL + "/mcp", ByURL: true}, "")
	s.Require().NoError(err)

	authorize, err := url.Parse(pending.AuthorizeURL)
	s.Require().NoError(err)
	s.Equal("notes.example", authorize.Host)
	s.Equal("notes-client", authorize.Query().Get("client_id"))
	s.Equal("notes.read notes.write", authorize.Query().Get("scope"))
	s.Equal(server.URL+"/mcp", authorize.Query().Get("resource"))
	s.Equal("S256", authorize.Query().Get("code_challenge_method"))
	s.NotEmpty(pending.CodeVerifier)
	s.Equal("https://notes.example/token", pending.TokenEndpoint)
}

// A config names any server it likes, so its login never takes a client the deployment
// registered for a plugin of the same name.
func (s *OAuthSuite) TestAServerNamedByURLNeverTakesTheDeploymentsClient() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://notes.example/authorize",
			TokenEndpoint:         "https://notes.example/token",
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	s.T().Setenv("NOTES_MCP_CLIENT_ID", "deployment-client")

	auth := &Auth{HTTP: server.Client(), PublicURL: "https://router.example"}
	_, err := auth.StartAuthorize(context.Background(),
		Plugin{ID: "notes", Name: "notes", URL: server.URL + "/mcp", Scopes: []string{"read"}, ByURL: true}, "")

	s.ErrorContains(err, "does not advertise dynamic client registration")
}

func (s *OAuthSuite) TestAServerWithALoginTheRouterCanMakePassesTheCheck() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-authorization-server", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(authServer{
			AuthorizationEndpoint: "https://notes.example/authorize",
			TokenEndpoint:         "https://notes.example/token",
			RegistrationEndpoint:  "http://" + r.Host + "/register",
		})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	auth := &Auth{HTTP: server.Client()}

	s.NoError(auth.CheckLogin(context.Background(), server.URL+"/mcp"))
}

func (s *OAuthSuite) TestAServerThatAnswersWithNoLoginFailsTheCheck() {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()

	err := (&Auth{HTTP: server.Client()}).CheckLogin(context.Background(), server.URL+"/mcp")

	var unreachable *url.Error
	s.ErrorContains(err, "oauth discovery")
	s.False(errors.As(err, &unreachable), "the server answered, so it is not merely unreachable")
}

func (s *OAuthSuite) TestAServerThatCannotBeReachedIsToldApartFromOneWithNoLogin() {
	server := httptest.NewServer(http.NotFoundHandler())
	server.Close()

	err := (&Auth{HTTP: server.Client()}).CheckLogin(context.Background(), server.URL+"/mcp")

	var unreachable *url.Error
	s.True(errors.As(err, &unreachable), err)
}

func (s *OAuthSuite) TestAServerPublishingWhereToLogInNeedsALogin() {
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/oauth-protected-resource/mcp", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(protectedResource{AuthorizationServers: []string{"https://notes.example"}})
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	needs, err := (&Auth{HTTP: server.Client()}).NeedsLogin(context.Background(), server.URL+"/mcp")

	s.Require().NoError(err)
	s.True(needs)
}

func (s *OAuthSuite) TestAServerRefusingARequestWithoutATokenNeedsALogin() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("WWW-Authenticate", `Bearer resource_metadata="https://notes.example/meta"`)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	needs, err := (&Auth{HTTP: server.Client()}).NeedsLogin(context.Background(), server.URL+"/mcp")

	s.Require().NoError(err)
	s.True(needs)
}

func (s *OAuthSuite) TestAServerAnsweringWithoutATokenNeedsNoLogin() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":0,"result":{}}`))
	}))
	defer server.Close()

	needs, err := (&Auth{HTTP: server.Client()}).NeedsLogin(context.Background(), server.URL+"/mcp")

	s.Require().NoError(err)
	s.False(needs)
}

func (s *OAuthSuite) TestWhetherAServerFailingOnItsSideNeedsALoginIsNotKnown() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "down", http.StatusBadGateway)
	}))
	defer server.Close()

	_, err := (&Auth{HTTP: server.Client()}).NeedsLogin(context.Background(), server.URL+"/mcp")

	s.ErrorContains(err, "502")
}

func (s *OAuthSuite) TestWhetherAServerThatCannotBeReachedNeedsALoginIsNotKnown() {
	server := httptest.NewServer(http.NotFoundHandler())
	server.Close()

	_, err := (&Auth{HTTP: server.Client()}).NeedsLogin(context.Background(), server.URL+"/mcp")

	var unreachable *url.Error
	s.True(errors.As(err, &unreachable), err)
}
