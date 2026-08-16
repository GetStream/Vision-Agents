package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

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
