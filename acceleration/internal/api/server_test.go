package api

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"gopkg.in/yaml.v3"

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

func (s *ServerSuite) TestVideoDefaultsRoundTripWithinSchemaBounds() {
	for _, requested := range []int{0, 2} {
		request := AgentConfigRequest{Name: "visual"}
		if requested != 0 {
			request.Video = &SessionVideo{MaxFrames: &requested}
		}
		response := agentConfigOf(storedConfig(request, "test"))
		s.Require().NotNil(response.Video)
		s.Require().NotNil(response.Video.MaxFrames)
		expected := requested
		if expected == 0 {
			expected = 1
		}
		s.Equal(expected, *response.Video.MaxFrames)
		replay := AgentConfigRequest{Name: response.Name, Video: response.Video}
		s.Equal(expected, storedConfig(replay, "test").VideoMaxFrames)
	}
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
	//
	// Proxy mode, because the contract under test is the one with every sort of caller
	// in it. The mode that asks for nothing has only backends, so a suite running under
	// it could not tell a refusal from an answer.
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
		Auth:    s.proxyAuth(),
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
}

// proxyAuth is the authenticator for a deployment behind something that has already
// worked out who the caller is.
func (s *ServerSuite) proxyAuth() auth.Authenticator {
	authenticator, err := auth.New(auth.Proxy, nil)
	s.Require().NoError(err)
	return authenticator
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

// A browser reaching a proxied deployment proves itself with a token rather than by naming
// a tenant, and a preflight turns any header it was not asked about into a blocked request.
// Allowing only the header a keyless deployment uses is how direct browser access breaks
// while looking like the origin was at fault.
func (s *ServerSuite) TestANamedOriginMaySendTheCredentialsAProxiedDeploymentWants() {
	allowed := s.origins("https://dash.example")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodOptions, "/v1/agents/sessions", nil)
	request.Header.Set("Origin", "https://dash.example")
	request.Header.Set("Access-Control-Request-Method", http.MethodPost)
	allowed.ServeHTTP(recorder, request)

	permitted := recorder.Header().Get("Access-Control-Allow-Headers")
	for _, header := range []string{"Authorization", auth.AuthTypeHeader, auth.APIKeyHeader, "Content-Type"} {
		s.Contains(permitted, header)
	}
}

func (s *ServerSuite) TestAPreflightIsAnsweredWithoutReachingAHandler() {
	allowed := s.origins("https://dash.example")

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodOptions, "/v1/agents/calls", nil)
	request.Header.Set("Origin", "https://dash.example")
	allowed.ServeHTTP(recorder, request)

	s.Equal(http.StatusNoContent, recorder.Code)
	// PUT is what replaces a live session's instructions, and a method missing from the
	// preflight is a request the browser never sends.
	s.Contains(recorder.Header().Get("Access-Control-Allow-Methods"), http.MethodPut)
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

// logging builds a handler that writes its log to the buffer it returns.
func (s *ServerSuite) logging() (http.Handler, *bytes.Buffer) {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	written := &bytes.Buffer{}
	server, err := NewServer(Options{
		Routers:     map[routing.Modality]routing.Inspector{routing.STT: speech},
		CORSOrigins: []string{"https://dash.example"},
		Auth:        s.proxyAuth(),
		Logger: slog.New(slog.NewTextHandler(written, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})),
	})
	s.Require().NoError(err)
	return server.Handler(), written
}

func (s *ServerSuite) TestARequestIsLoggedWithWhatItAskedForAndWhatItGot() {
	handler, written := s.logging()

	request := httptest.NewRequest(http.MethodGet, "/v1/tts/providers", nil)
	request.Header.Set(CustomerHeader, "acme")
	handler.ServeHTTP(httptest.NewRecorder(), request)

	logged := written.String()
	s.Contains(logged, "method=GET")
	s.Contains(logged, "path=/v1/tts/providers")
	s.Contains(logged, "status=404", "the status the caller was told, not the one it hoped for")
	s.Contains(logged, "customer=acme")
	s.Regexp(`duration=[0-9]`, logged)
}

func (s *ServerSuite) TestALoggedRequestDoesNotRepeatTheQuery() {
	handler, written := s.logging()

	// A socket names its customer in the query and a vendor names a token, so the log
	// records the path alone.
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers?customer_id=acme", nil)
	handler.ServeHTTP(httptest.NewRecorder(), request)

	logged := written.String()
	s.Contains(logged, "path=/v1/stt/providers")
	s.NotContains(logged, "customer_id=acme")
}

func (s *ServerSuite) TestARefusedServerSideOperationIsLoggedAsForbidden() {
	handler, written := s.logging()

	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs", strings.NewReader(`{}`))
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)
	handler.ServeHTTP(httptest.NewRecorder(), request)

	s.Contains(written.String(), "status=403",
		"a refusal is a logged answer, not a request that never arrived")
}

func (s *ServerSuite) TestAPreflightIsNotLogged() {
	handler, written := s.logging()

	request := httptest.NewRequest(http.MethodOptions, "/v1/agents/calls", nil)
	request.Header.Set("Origin", "https://dash.example")
	handler.ServeHTTP(httptest.NewRecorder(), request)

	s.Empty(written.String(), "asking permission is not a request worth a line")
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

func (s *ServerSuite) TestRoutesOfferTheTitledShortcutsWithTheModelsTheyResolveTo() {
	recorder := s.get("/v1/stt/routes", "acme")

	s.Equal(http.StatusOK, recorder.Code)
	var routes []Route
	s.decode(recorder, &routes)
	s.Require().NotEmpty(routes)
	s.Equal("en-low-latency", routes[0].Id, "the default comes first")
	s.Equal("Fast English", routes[0].Title)
	s.NotEmpty(routes[0].Description)
	s.Require().NotEmpty(routes[0].Candidates)
	s.Equal("deepgram", routes[0].Candidates[0].Provider)
	for _, route := range routes {
		s.NotEqual("en-recorded", route.Id, "an untitled shortcut is not offered")
	}
}

func (s *ServerSuite) TestProvidersHaveNoShareWithoutStatistics() {
	recorder := s.get("/v1/stt/providers", "acme")

	var providers []Provider
	s.decode(recorder, &providers)
	s.Require().NotEmpty(providers)
	s.Require().NotNil(providers[0].UsageShare)
	s.Zero(*providers[0].UsageShare)
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

func (s *ServerSuite) TestSpendRequiresTheCustomerHeader() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/spend?from="+from+"&to="+to, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestSpendRejectsAnInvertedWindow() {
	from := time.Now().UTC().Format(time.RFC3339)
	to := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/spend?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "to must be after from")
}

func (s *ServerSuite) TestSpendRejectsATagFilterThatIsNotAPair() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/spend?from="+from+"&to="+to+"&tag=support", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "must be written key:value")
}

func (s *ServerSuite) TestSpendRejectsAGroupThatCannotBeCharted() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/spend?from="+from+"&to="+to+"&limit=0", "acme")

	s.Equal(http.StatusBadRequest, recorder.Code, "the spec makes one the smallest limit")
}

func (s *ServerSuite) TestTagKeysRequireTheCustomerHeader() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/tags/keys?from="+from+"&to="+to, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestTagKeysReportWhenNoDatabaseIsConfigured() {
	from := time.Now().Add(-time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/tags/keys?from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "no database configured")
}

func (s *ServerSuite) TestActivityRequiresTheCustomerHeader() {
	from := time.Now().Add(-24 * time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/activity?from="+from+"&to="+to, "")

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestActivityRejectsAGranularityItCannotCountUsersOver() {
	from := time.Now().Add(-24 * time.Hour).UTC().Format(time.RFC3339)
	to := time.Now().UTC().Format(time.RFC3339)

	recorder := s.get("/v1/stats/activity?granularity=hourly&from="+from+"&to="+to, "acme")

	s.Equal(http.StatusBadRequest, recorder.Code,
		"distinct users cannot be summed, so the hours the spend paths take are not offered here")
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

// speech is a speech-to-text router over the default configuration.
func (s *ServerSuite) speech() routing.Inspector {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)
	return speech
}

// keyed builds a handler in api_key mode where one key resolves to one app.
func (s *ServerSuite) keyed(key, secret string) http.Handler {
	return s.serving(s.speech(), s.keyAuth(key, auth.App{
		OrganizationID: "org-1", AppID: "app-1", Secret: secret,
	}))
}

// keyAuth is an api_key authenticator over a single key belonging to one app.
func (s *ServerSuite) keyAuth(key string, app auth.App) auth.Authenticator {
	authenticator, err := auth.New(auth.APIKey, func(_ context.Context, presented string) (auth.App, error) {
		if presented != key {
			return auth.App{}, auth.ErrUnauthenticated
		}
		return app, nil
	})
	s.Require().NoError(err)
	return authenticator
}

// serving builds a handler over one router and one way of deciding who a caller is.
func (s *ServerSuite) serving(speech routing.Inspector, authenticator auth.Authenticator) http.Handler {
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: speech},
	}, WithAuthenticator(authenticator))
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

// serverToken signs a token a backend mints for itself, which names no user.
func (s *ServerSuite) serverToken(secret string) string {
	signed, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"server": true,
		"exp":    time.Now().Add(time.Hour).Unix(),
	}).SignedString([]byte(secret))
	s.Require().NoError(err)
	return signed
}

// asUser and asBackend credential a request the two ways a caller can be credentialed.
func (s *ServerSuite) asUser(r *http.Request, key, secret string) *http.Request {
	r.Header.Set(auth.APIKeyHeader, key)
	r.Header.Set("Authorization", "Bearer "+s.token(secret))
	r.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)
	return r
}

func (s *ServerSuite) asBackend(r *http.Request, key, secret string) *http.Request {
	r.Header.Set(auth.APIKeyHeader, key)
	r.Header.Set("Authorization", "Bearer "+s.serverToken(secret))
	r.Header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)
	return r
}

func (s *ServerSuite) TestAProxyNamesTheCustomerAndItsOrganization() {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	request.Header.Set(auth.AppHeader, "app-1")
	request.Header.Set(auth.OrganizationHeader, "org-1")
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusOK, recorder.Code)
}

func (s *ServerSuite) TestAnAppTurnsAwayALevelWithAForbidden() {
	// A level an app refuses is not a caller that gets a narrower API, it is a caller
	// that does not get in, so it is answered at the door and for every path alike.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.serving(s.speech(), s.keyAuth(key, auth.App{
		OrganizationID: "org-1", AppID: "app-1", Secret: secret,
		Levels: auth.Levels{NoAnonymous: true},
	}))

	// A token naming no user is an anonymous caller, however good its signature.
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/agents/sessions", nil)
	handler.ServeHTTP(recorder, s.asUser(request, key, secret))

	s.Equal(http.StatusForbidden, recorder.Code)
	s.Contains(recorder.Body.String(), "level of user",
		"a caller that has proved who it is should be told what the problem is")
}

func (s *ServerSuite) TestAnAppTakesTheLevelsItHasNotTurnedAway() {
	// The other half: the default admits, so an app with no settings written is not one
	// whose users have all been locked out.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/agents/sessions", nil)
	handler.ServeHTTP(recorder, s.asUser(request, key, secret))

	s.NotEqual(http.StatusForbidden, recorder.Code)
}

func (s *ServerSuite) TestADeploymentCanAnswerForItself() {
	// The custom mode. Nothing about who a caller is has to come from this package: a
	// deployment embedding it supplies the whole answer and the rest of the chain —
	// server-side, ownership, quota — reads it the same as any other.
	handler := s.serving(s.speech(), auth.Func(
		func(context.Context, *http.Request) (auth.Principal, error) {
			return auth.Principal{AppID: "from-the-embedder", Kind: auth.KindServer, ServerSide: true}, nil
		}))

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil))

	s.Equal(http.StatusOK, recorder.Code, "a caller presenting nothing at all was admitted")
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
	// As a backend, because listing providers is server-side only like everything the
	// spec does not open. What is under test here is that the credential is accepted.
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	handler.ServeHTTP(recorder, s.asBackend(request, key, secret))

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

func (s *ServerSuite) TestAUsersDeviceMayNotConfigureAnAgent() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, s.asUser(request, key, secret))

	s.Equal(http.StatusForbidden, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "server-side only")
}

func (s *ServerSuite) TestABackendMayConfigureAnAgent() {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, s.asBackend(request, key, secret))

	// It reaches the handler, which refuses it for a reason of its own: this deployment
	// has no database to keep a config in. What matters is that it got that far.
	s.Equal(http.StatusBadRequest, recorder.Code)
}

func (s *ServerSuite) TestAUsersDeviceMayNotReadTheAgentsItTalksTo() {
	// Reading a config is server-side only like writing one. What an app shows about the
	// agent is what its own backend chose to tell it, not what it can ask this service.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/agents/configs", nil)
	handler.ServeHTTP(recorder, s.asUser(request, key, secret))

	s.Equal(http.StatusForbidden, recorder.Code)
}

func (s *ServerSuite) TestAUsersDeviceMayNotWaitForOtherPeoplesCalls() {
	// The dispatch socket is offered other people's callers, so anything that can open
	// one can answer for the whole app.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/dispatch", nil)
	handler.ServeHTTP(recorder, s.asUser(request, key, secret))

	s.Equal(http.StatusForbidden, recorder.Code)

	var failure Error
	s.decode(recorder, &failure)
	s.Contains(failure.Error, "server-side only")
}

func (s *ServerSuite) TestACallerWithNoCredentialIsToldToAuthenticateFirst() {
	// A 403 on a request that never said who it was would send the caller looking for a
	// permission problem instead of the missing credential.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/agents/configs", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ServerSuite) TestEveryOperationTheSpecDoesNotOpenIsRefusedToAUsersDevice() {
	// The middleware reads the spec, so this is what proves the default reaches every
	// operation rather than only the ones a test happened to name. It is the whole point
	// of the inverted default: an operation nobody thought about is refused here.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	spec, err := GetSpec()
	s.Require().NoError(err)

	refused := 0
	for path, item := range spec.Paths.Map() {
		for method, operation := range item.Operations() {
			if operation.Security != nil && len(*operation.Security) == 0 {
				continue
			}
			if open, ok := operation.Extensions[clientAccessibleExtension].(bool); ok && open {
				continue
			}
			refused++

			// A path parameter is filled with anything: the refusal comes before the
			// handler that would look the resource up.
			target := regexp.MustCompile(`\{[^}]+\}`).ReplaceAllString(path, "x")
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(method, target, strings.NewReader(`{}`))
			request.Header.Set("Content-Type", "application/json")
			handler.ServeHTTP(recorder, s.asUser(request, key, secret))

			s.Equal(http.StatusForbidden, recorder.Code, method+" "+path)
		}
	}
	s.NotZero(refused, "the spec leaves nothing server-side only")
}

func (s *ServerSuite) TestEveryOperationTheSpecOpensIsReachableByAUsersDevice() {
	// The other half. A default that refused everything would pass the test above and
	// leave nobody able to hold a conversation.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	spec, err := GetSpec()
	s.Require().NoError(err)

	opened := 0
	for path, item := range spec.Paths.Map() {
		for method, operation := range item.Operations() {
			if open, ok := operation.Extensions[clientAccessibleExtension].(bool); !ok || !open {
				continue
			}
			opened++

			target := regexp.MustCompile(`\{[^}]+\}`).ReplaceAllString(path, "x")
			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(method, target, strings.NewReader(`{}`))
			request.Header.Set("Content-Type", "application/json")
			handler.ServeHTTP(recorder, s.asUser(request, key, secret))

			s.NotEqual(http.StatusForbidden, recorder.Code, method+" "+path)
		}
	}
	s.NotZero(opened, "the spec opens nothing to a client")
}

func (s *ServerSuite) TestTheRoutesLeftOutOfTheSpecAreStillDecidedOneWayOrTheOther() {
	// Excluding an operation from generation drops it from the embedded spec the two
	// tests above read, which is the one way an inverted default can fail open: nothing
	// refuses what nothing can see. So unspecifiedRoutes has to name every excluded
	// operation, not merely be right about the ones it happens to name.
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	handler := s.keyed(key, secret)

	for route, open := range excludedRoutes(s.T()) {
		s.Contains(unspecifiedRoutes, route,
			"%s is excluded from generation, so the middleware cannot see it", route)
		s.Equal(open, unspecifiedRoutes[route], "%s is open in the spec but not here", route)

		method, path, found := strings.Cut(route, " ")
		s.Require().True(found, route)
		target := regexp.MustCompile(`\{[^}]+\}`).ReplaceAllString(path, "x")
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(method, target, nil)
		handler.ServeHTTP(recorder, s.asUser(request, key, secret))

		if open {
			s.NotEqual(http.StatusForbidden, recorder.Code, route)
		} else {
			s.Equal(http.StatusForbidden, recorder.Code, route)
		}
	}
}

// excludedRoutes reads the operations kept out of generation, as routes and whether each
// is open to a client.
//
// Both files are read off disk rather than from the embedded spec, because what is being
// checked is the very thing the embedded spec is missing: the generator's exclude list on
// one side and the operations it names on the other.
func excludedRoutes(t *testing.T) map[string]bool {
	t.Helper()

	var codegen struct {
		OutputOptions struct {
			Excluded []string `yaml:"exclude-operation-ids"`
		} `yaml:"output-options"`
	}
	read(t, "../../api/oapi-codegen.yaml", &codegen)

	var spec struct {
		Paths map[string]map[string]struct {
			OperationID string `yaml:"operationId"`
			Open        bool   `yaml:"x-client-accessible"`
			Security    *[]map[string][]string
		} `yaml:"paths"`
	}
	read(t, "../../api/openapi.yaml", &spec)

	excluded := map[string]bool{}
	for _, id := range codegen.OutputOptions.Excluded {
		excluded[id] = false
	}

	routes := map[string]bool{}
	for path, item := range spec.Paths {
		for method, operation := range item {
			if _, ok := excluded[operation.OperationID]; !ok {
				continue
			}
			free := operation.Security != nil && len(*operation.Security) == 0
			routes[strings.ToUpper(method)+" "+path] = operation.Open || free
			excluded[operation.OperationID] = true
		}
	}
	for id, found := range excluded {
		require.True(t, found, "%s is excluded from generation but is not in the spec", id)
	}
	return routes
}

func read(t *testing.T, path string, into any) {
	t.Helper()

	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	require.NoError(t, yaml.Unmarshal(raw, into))
}

// forwarded is a request from proxyAddr carrying an X-Forwarded-For chain.
func (s *ServerSuite) forwarded(proxyAddr string, chain ...string) *http.Request {
	request := httptest.NewRequest(http.MethodGet, "/v1/stt/providers", nil)
	request.RemoteAddr = proxyAddr
	for _, entry := range chain {
		request.Header.Add(ForwardedHeader, entry)
	}
	return request
}

// trusted parses ranges the way the router does at startup.
func (s *ServerSuite) trusted(ranges ...string) []netip.Prefix {
	parsed, err := TrustedProxies(ranges)
	s.Require().NoError(err)
	return parsed
}

func (s *ServerSuite) TestWithNoTrustedProxyTheConnectionIsTheCaller() {
	// Nothing vouches for the header, so it is not read at all.
	request := s.forwarded("198.51.100.7:44321", "203.0.113.9")

	s.Equal("198.51.100.7", clientIP(request, nil))
}

func (s *ServerSuite) TestATrustedProxyNamesTheCallerBehindIt() {
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")

	s.Equal("203.0.113.9", clientIP(request, s.trusted("10.0.0.0/8")))
}

func (s *ServerSuite) TestAForwardedEntryTheCallerWroteThemselvesIsIgnored() {
	// The caller prepended a victim's address hoping to spend their allowance. Only the
	// rightmost entry was written by our own proxy, so that is the one believed.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.250, 203.0.113.9")

	s.Equal("203.0.113.9", clientIP(request, s.trusted("10.0.0.0/8")))
}

func (s *ServerSuite) TestTheCallerIsFoundThroughSeveralOfOurOwnProxies() {
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9, 10.0.0.9", "10.0.0.7")

	s.Equal("203.0.113.9", clientIP(request, s.trusted("10.0.0.0/8")))
}

func (s *ServerSuite) TestAChainOfOnlyOurOwnProxiesFallsBackToTheConnection() {
	request := s.forwarded("10.0.0.5:44321", "10.0.0.9")

	s.Equal("10.0.0.5", clientIP(request, s.trusted("10.0.0.0/8")))
}

func (s *ServerSuite) TestAnUnreadableForwardedEntryStopsTheWalk() {
	// There is no telling whose entry sits left of a broken one, so the proxy is as far as
	// the chain can be trusted.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9, nonsense")

	s.Equal("10.0.0.5", clientIP(request, s.trusted("10.0.0.0/8")))
}

func (s *ServerSuite) TestTrustedProxiesTakesABareAddressAsWellAsARange() {
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")

	s.Equal("203.0.113.9", clientIP(request, s.trusted("10.0.0.5")))
}

func (s *ServerSuite) TestTrustedProxiesRefusesSomethingThatIsNotARange() {
	_, err := TrustedProxies([]string{"not-a-range"})

	s.Require().Error(err)
	s.Contains(err.Error(), "not-a-range")
}

// callerSeenBy runs a request through withCustomer and reports what reached the handler.
func (s *ServerSuite) callerSeenBy(server *Server, request *http.Request) routing.Caller {
	var seen routing.Caller
	handler := server.withCustomer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		seen = CallerFrom(r.Context())
	}))
	handler.ServeHTTP(httptest.NewRecorder(), request)
	return seen
}

// contextSeenBy runs a request through withCustomer and returns the context the handler
// was given, for the tests that read more than one thing off it.
func (s *ServerSuite) contextSeenBy(server *Server, request *http.Request) context.Context {
	seen := context.Background()
	handler := server.withCustomer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		seen = r.Context()
	}))
	handler.ServeHTTP(httptest.NewRecorder(), request)
	return seen
}

// proxiedServer is a server in proxy mode that believes one range of proxies.
func (s *ServerSuite) proxiedServer() *Server {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	server, err := NewServer(Options{
		Routers:        map[routing.Modality]routing.Inspector{routing.STT: speech},
		Auth:           s.proxyAuth(),
		TrustedProxies: s.trusted("10.0.0.0/8"),
	})
	s.Require().NoError(err)
	return server
}

func (s *ServerSuite) TestAnEndUserIsNamedAndPlacedForTheLimitToCount() {
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.UserHeader, "user-1")
	request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)

	seen := s.callerSeenBy(s.proxiedServer(), request)

	s.Equal(routing.Caller{UserID: "user-1", IP: "203.0.113.9"}, seen)
}

func (s *ServerSuite) TestABackendSaysWhichUserItIsActingFor() {
	// The user a backend names is what the session it opens will belong to, so it is
	// carried rather than dropped. What keeps the day's limit off it is that the caller
	// is server-side, not that there is no caller: see TestABackendIsCountedAgainstNobody.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.UserHeader, "user-1")

	ctx := s.contextSeenBy(s.proxiedServer(), request)

	s.Equal("user-1", CallerFrom(ctx).UserID)
	s.True(ServerSideFrom(ctx))
	s.Equal(auth.KindServer, KindFrom(ctx))
}

func (s *ServerSuite) TestABackendIsCountedAgainstNobody() {
	// A process the customer runs is trusted with its own spend, even when it has said
	// whose behalf it is working on: that name is a user the customer chose to do work
	// for, and charging their day for it would be charging them for their own backend.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.UserHeader, "user-1")

	ctx := s.contextSeenBy(s.proxiedServer(), request)

	s.True(exemptFromQuota(ctx))
}

func (s *ServerSuite) TestAnEndUserOfTheSameNameIsNotExempt() {
	// The other half: the exemption is about what the caller is, not what it is called,
	// so naming yourself the same user a backend would is no way out of the limit.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.UserHeader, "user-1")
	request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)

	ctx := s.contextSeenBy(s.proxiedServer(), request)

	s.False(exemptFromQuota(ctx))
}

func (s *ServerSuite) TestAnEndUserWithNoNameIsStillCountedByAddress() {
	// A token that names no user is still a browser somewhere, and the address is what is
	// left to count against.
	request := s.forwarded("10.0.0.5:44321", "203.0.113.9")
	request.Header.Set(CustomerHeader, "acme")
	request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeJWT)

	seen := s.callerSeenBy(s.proxiedServer(), request)

	s.Equal(routing.Caller{IP: "203.0.113.9"}, seen)
	s.False(seen.Anonymous())
}

func (s *ServerSuite) TestAnUnauthenticatedRequestCarriesNoCaller() {
	seen := s.callerSeenBy(s.proxiedServer(), s.forwarded("10.0.0.5:44321", "203.0.113.9"))

	s.True(seen.Anonymous())
}
