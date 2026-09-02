package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
)

type ServerSuite struct {
	suite.Suite
	handler http.Handler
}

func TestServerSuite(t *testing.T) {
	suite.Run(t, new(ServerSuite))
}

func (s *ServerSuite) SetupTest() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	// Only speech-to-text is wired, so the text-to-speech paths exercise the 404 an
	// unserved modality gets. No store and no live client: this suite covers the HTTP
	// contract, so the endpoints that need a database report that rather than being
	// exercised here.
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
}

// get issues a request, optionally with the customer header.
func (s *ServerSuite) get(path, customerID string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(http.MethodGet, path, nil)
	if customerID != "" {
		request.Header.Set(CustomerHeader, customerID)
	}
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}

func (s *ServerSuite) decode(recorder *httptest.ResponseRecorder, target any) {
	s.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), target))
}

func (s *ServerSuite) TestNewServerRequiresARouter() {
	_, err := NewServer(Options{})
	s.ErrorContains(err, "at least one router is required")
}

func (s *ServerSuite) TestASocketMayNameItsCustomerInTheQuery() {
	// The browser WebSocket API cannot set a header, and a dashboard watching a live call
	// is exactly the caller that has to open one.
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers?customer_id=acme", nil)
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *ServerSuite) TestABrowserIsTurnedAwayUnlessItsOriginWasNamed() {
	allowed := s.origins("https://dash.example")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	request.Header.Set("Origin", "https://somewhere.else")
	allowed.ServeHTTP(recorder, request)

	s.Empty(recorder.Header().Get("Access-Control-Allow-Origin"),
		"an origin nobody named is a browser that gets nothing back")
}

func (s *ServerSuite) TestANamedOriginMayReadTheApiAndSendTheCustomerHeader() {
	allowed := s.origins("https://dash.example")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	request.Header.Set("Origin", "https://dash.example")
	allowed.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
	s.Equal("https://dash.example", recorder.Header().Get("Access-Control-Allow-Origin"))
	s.Contains(recorder.Header().Get("Access-Control-Allow-Headers"), CustomerHeader)
}

func (s *ServerSuite) TestAPreflightIsAnsweredWithoutReachingAHandler() {
	allowed := s.origins("https://dash.example")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodOptions, "/v1/agents/calls", nil)
	request.Header.Set("Origin", "https://dash.example")
	allowed.ServeHTTP(recorder, request)

	s.Equal(http.StatusNoContent, recorder.Code)
	s.Contains(recorder.Header().Get("Access-Control-Allow-Methods"), http.MethodPatch)
}

func (s *ServerSuite) TestWithoutNamedOriginsNoBrowserIsLetIn() {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	request.Header.Set("Origin", "https://dash.example")
	s.handler.ServeHTTP(recorder, request)

	s.Empty(recorder.Header().Get("Access-Control-Allow-Origin"))
}

// origins builds a handler that lets the named browser origins in.
func (s *ServerSuite) origins(allowed ...string) http.Handler {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers:     map[routing.Modality]routing.Inspector{routing.STT: speech},
		CORSOrigins: allowed,
	})
	s.Require().NoError(err)
	return server.Handler()
}

func (s *ServerSuite) TestHealthNeedsNoCustomerHeader() {
	recorder := s.get("/health", "")

	s.Equal(http.StatusOK, recorder.Code)

	var status HealthStatus
	s.decode(recorder, &status)
	s.Equal(Ok, status.Status)
	s.Equal("not configured", status.Dependencies["postgres"])
	s.Equal("not configured", status.Dependencies["redis"])
	s.Equal("ok", status.Dependencies["stt"], "health reports which modalities are served")
	s.NotContains(status.Dependencies, "tts")
}

func (s *ServerSuite) TestProvidersRequireTheCustomerHeader() {
	recorder := s.get("/v1/stt/providers", "")

	s.Equal(http.StatusUnauthorized, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, CustomerHeader)
}

func (s *ServerSuite) TestProvidersListTheConfiguredCapabilities() {
	recorder := s.get("/v1/stt/providers", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var providers []Provider
	s.decode(recorder, &providers)
	s.Require().NotEmpty(providers)

	names := map[string]Provider{}
	for _, provider := range providers {
		names[provider.Provider+"/"+provider.Model] = provider
	}
	s.Contains(names, "deepgram/flux-general-en")
	s.Contains(names, "parakeet/parakeet-tdt-0.6b-v3")

	english := names["deepgram/flux-general-en"]
	s.Equal([]string{"en"}, english.Languages)
	s.True(english.Realtime)
	s.Equal(LowLatency, english.Tier)
	s.True(english.Health.Available, "an unmeasured provider is available")
}

func (s *ServerSuite) TestAnUnservedModalityIsNotFound() {
	recorder := s.get("/v1/tts/providers", "acme")

	s.Equal(http.StatusNotFound, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "does not route tts")
}

func (s *ServerSuite) TestAModalityTheRouterHasNeverHeardOfIsNotFound() {
	recorder := s.get("/v1/llm/providers", "acme")

	s.Equal(http.StatusNotFound, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "does not route llm")
}

func (s *ServerSuite) TestResolveReturnsCandidatesBestFirst() {
	recorder := s.get("/v1/stt/routes/en-low-latency", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var candidates []Candidate
	s.decode(recorder, &candidates)
	s.Require().NotEmpty(candidates)
	s.Equal("deepgram", candidates[0].Provider)
	s.Equal("flux-general-en", candidates[0].Model)
}

func (s *ServerSuite) TestResolveNarrowsOnLanguageHints() {
	recorder := s.get("/v1/stt/routes/multilingual-low-latency?language=de", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var candidates []Candidate
	s.decode(recorder, &candidates)
	s.Require().NotEmpty(candidates)
	for _, candidate := range candidates {
		s.NotEqual("flux-general-en", candidate.Model, "the English model cannot serve German")
	}
}

func (s *ServerSuite) TestResolveRejectsAnUnknownTarget() {
	recorder := s.get("/v1/stt/routes/does-not-exist", "acme")

	s.Equal(http.StatusNotFound, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "unknown target")
}

func (s *ServerSuite) TestResolveRejectsAnUnservableLanguage() {
	recorder := s.get("/v1/stt/routes/en-low-latency?language=tlh", "acme")

	s.Equal(http.StatusNotFound, recorder.Code)
}

func (s *ServerSuite) TestStatsRequireTheCustomerHeader() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats?from="+from+"&to="+to, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestStatsReportWhenNoDatabaseIsConfigured() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database configured")
}

func (s *ServerSuite) TestStatsRejectAnInvertedWindow() {
	from := time.Now().UTC().Format(time.RFC3339)
	to := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "to must be after from")
}

func (s *ServerSuite) TestStatsRequireAWindow() {
	recorder := s.get("/v1/stt/stats", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code, "the spec makes from and to required")
}

func (s *ServerSuite) TestStatsRejectATagFilterThatIsNotAPair() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats?from="+from+"&to="+to+"&tag=moderation", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "must be written key:value")
}

func (s *ServerSuite) TestTagStatsRequireAKeyToGroupBy() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats/tags?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code, "the spec makes key required")
}

func (s *ServerSuite) TestTagStatsRequireTheCustomerHeader() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stt/stats/tags?key=project&from="+from+"&to="+to, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestStatsAreServedForModalitiesThatAreRecordedButNotRouted() {
	// Nothing routes memory, but it is recorded against the customer and costs them
	// money, so asking what it cost is not a 404.
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/memory/stats?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database configured", "the modality was accepted, the store is what is missing")
}

func (s *ServerSuite) TestRollupRequiresTheCustomerHeader() {
	body := `{"from":"2026-03-01T00:00:00Z","to":"2026-03-02T00:00:00Z"}`
	request := httptest.NewRequest(http.MethodPost, "/v1/stats/rollup", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestRollupReportsWhenNoDatabaseIsConfigured() {
	body := `{"from":"2026-03-01T00:00:00Z","to":"2026-03-02T00:00:00Z"}`
	request := httptest.NewRequest(http.MethodPost, "/v1/stats/rollup", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database configured")
}

func (s *ServerSuite) TestRollupRejectsAnInvertedWindow() {
	body := `{"from":"2026-03-02T00:00:00Z","to":"2026-03-01T00:00:00Z"}`
	request := httptest.NewRequest(http.MethodPost, "/v1/stats/rollup", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()

	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "to must be after from")
}

func (s *ServerSuite) TestPluginsAreListedFromTheCatalog() {
	recorder := s.get("/v1/agents/plugins", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var listed []Plugin
	s.decode(recorder, &listed)
	s.Len(listed, 5)
	s.Equal("slack", listed[0].Id)
}

func (s *ServerSuite) TestPluginSearchFiltersTheCatalog() {
	recorder := s.get("/v1/agents/plugins?q=cal", "acme")

	s.Equal(http.StatusOK, recorder.Code)

	var listed []Plugin
	s.decode(recorder, &listed)
	s.Len(listed, 2)
	s.Equal("calendly", listed[0].Id)
	s.Equal("calcom", listed[1].Id)
}

func (s *ServerSuite) TestShopifyAuthorizeWithoutAnInstanceIsRefused() {
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs/cfg/plugins/shopify/authorize", strings.NewReader(`{}`))
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "needs")
}

func (s *ServerSuite) TestAnUnknownPluginIsRefused() {
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs/cfg/plugins/notion/authorize", strings.NewReader(`{}`))
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusBadRequest, recorder.Code)
}

// keyed builds a handler in api_key mode where one key resolves to one app.
func (s *ServerSuite) keyed(key, secret string) http.Handler {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		if presented != key {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return auth.App{OrganizationID: "org-1", AppID: "app-1", Secret: secret}, nil
	})
	s.Require().NoError(err)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
		Auth:    authenticator,
	})
	s.Require().NoError(err)
	return server.Handler()
}

// token signs a caller's token with an app secret.
func (s *ServerSuite) token(secret string) string {
	signed, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
	}).SignedString([]byte(secret))
	s.Require().NoError(err)
	return signed
}

func (s *ServerSuite) TestAProxyNamesTheCustomerAndItsOrganization() {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	request.Header.Set(auth.AppHeader, "app-1")
	request.Header.Set(auth.OrganizationHeader, "org-1")
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *ServerSuite) TestAKeyedDeploymentIgnoresTheHeadersAProxyWouldSet() {
	// Without this the mode is theatre: anyone could skip the key by naming themselves the
	// way the trusted proxy would.
	handler := s.keyed("vak_live_0123456789abcdef00000000", "vas_live_s3cret")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	request.Header.Set(auth.AppHeader, "app-1")
	request.Header.Set(CustomerHeader, "app-1")
	request.Header.Set(auth.OrganizationHeader, "org-1")
	handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestAKeyedDeploymentAcceptsAKeyAndItsToken() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	request.Header.Set(auth.APIKeyHeader, key)
	request.Header.Set("Authorization", "Bearer "+s.token(secret))
	handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *ServerSuite) TestEveryAuthenticationFailureLooksTheSame() {
	// Telling a caller that the key was real but the token was not is a free way to find
	// out which keys exist, so all four answers have to be one answer.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	attempts := map[string]func(*http.Request){
		"nothing at all": func(*http.Request) {},
		"an unknown key": func(r *http.Request) {
			r.Header.Set(auth.APIKeyHeader, "vak_live_ffffffffffffffff00000000")
			r.Header.Set("Authorization", "Bearer "+s.token(secret))
		},
		"a malformed key": func(r *http.Request) {
			r.Header.Set(auth.APIKeyHeader, "nonsense")
			r.Header.Set("Authorization", "Bearer "+s.token(secret))
		},
		"a token signed with the wrong secret": func(r *http.Request) {
			r.Header.Set(auth.APIKeyHeader, key)
			r.Header.Set("Authorization", "Bearer "+s.token("vas_live_wrong"))
		},
	}

	var bodies []string
	for name, attempt := range attempts {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
		attempt(request)
		handler.ServeHTTP(recorder, request)

		s.Equal(http.StatusUnauthorized, recorder.Code, name)
		bodies = append(bodies, recorder.Body.String())
	}
	for _, body := range bodies {
		s.Equal(bodies[0], body)
	}
}

func (s *ServerSuite) TestHealthStaysReachableWithoutACredential() {
	// A liveness probe holds no API key, and neither does the vendor fetching a call plan.
	handler := s.keyed("vak_live_0123456789abcdef00000000", "vas_live_s3cret")

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/health", nil))

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *ServerSuite) TestASocketRefusesAnOriginThatWasNotNamed() {
	// The upgrade is the way around CORS if it accepts every origin, since the browser
	// sends the cookies either way.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers:     map[routing.Modality]routing.Inspector{routing.STT: speech},
		Dispatch:    dispatch.NewPool(),
		CORSOrigins: []string{"https://dash.example"},
	})
	s.Require().NoError(err)
	handler := server.Handler()

	handshake := func(origin string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "/v1/dispatch?customer_id=acme", nil)
		request.Header.Set("Origin", origin)
		request.Header.Set("Connection", "Upgrade")
		request.Header.Set("Upgrade", "websocket")
		request.Header.Set("Sec-WebSocket-Version", "13")
		request.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
		handler.ServeHTTP(recorder, request)
		return recorder
	}

	s.Equal(http.StatusForbidden, handshake("https://evil.example").Code)

	// A named origin gets past the check and fails further in, because a recorder cannot
	// be hijacked into a socket. What matters is that it was not turned away here.
	s.NotEqual(http.StatusForbidden, handshake("https://dash.example").Code)
}

func (s *ServerSuite) TestASocketAcceptsACallerThatNamesNoOrigin() {
	// A server-to-server client sends no Origin, and there is no browser session for
	// another site to ride on, so there is nothing for the check to protect against.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers:     map[routing.Modality]routing.Inspector{routing.STT: speech},
		Dispatch:    dispatch.NewPool(),
		CORSOrigins: []string{"https://dash.example"},
	})
	s.Require().NoError(err)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/dispatch?customer_id=acme", nil)
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Sec-WebSocket-Version", "13")
	request.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	server.Handler().ServeHTTP(recorder, request)

	s.NotEqual(http.StatusForbidden, recorder.Code)
}
