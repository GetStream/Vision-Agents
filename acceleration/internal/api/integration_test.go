//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

type APIIntegrationSuite struct {
	suite.Suite
	ctx        context.Context
	store      *store.Store
	live       *live.Client
	server     *httptest.Server
	customerID string
	base       time.Time
}

func TestAPIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(APIIntegrationSuite))
}

func (s *APIIntegrationSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || address == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}

	s.ctx = context.Background()

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(pgStore.Migrate(s.ctx))
	s.store = pgStore

	liveClient, err := live.New(live.Options{Address: address})
	s.Require().NoError(err)
	s.live = liveClient

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	speech, err := sttrouter.New(sttrouter.Options{
		Config:   config[routing.STT],
		Registry: sttrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speech.Close)

	voice, err := ttsrouter.New(ttsrouter.Options{
		Config:   config[routing.TTS],
		Registry: ttsrouter.DefaultRegistry(),
		Store:    pgStore,
		Live:     liveClient,
	})
	s.Require().NoError(err)
	s.T().Cleanup(voice.Close)

	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{
			routing.STT: speech,
			routing.TTS: voice,
		},
		Store: pgStore,
		Live:  liveClient,
	})
	s.Require().NoError(err)
	s.server = httptest.NewServer(server.Handler())

	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

func (s *APIIntegrationSuite) TearDownSuite() {
	if s.server != nil {
		s.server.Close()
	}
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
	if s.live != nil {
		s.live.Close()
	}
}

func (s *APIIntegrationSuite) SetupTest() {
	s.customerID = "customer-" + time.Now().Format("150405.000000000")
}

// do issues a request against the live test server with the customer header set.
func (s *APIIntegrationSuite) do(method, path, body string) (*http.Response, []byte) {
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}

	request, err := http.NewRequestWithContext(s.ctx, method, s.server.URL+path, reader)
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, s.customerID)
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	payload := make([]byte, 0)
	buffer := make([]byte, 4096)
	for {
		n, err := response.Body.Read(buffer)
		payload = append(payload, buffer[:n]...)
		if err != nil {
			break
		}
	}
	return response, payload
}

// recordTurn stores a completed speech-to-text turn for the current customer.
func (s *APIIntegrationSuite) recordTurn(at time.Time, audioMs int64, latencyMs float64, success bool) {
	request := &store.Request{
		Modality:   "stt",
		CustomerID: s.customerID,
		Provider:   "deepgram",
		Model:      "flux-general-en",
		StartedAt:  at,
		AudioMs:    audioMs,
		LatencyMs:  &latencyMs,
		Success:    success,
	}
	if !success {
		request.ErrorCode = "provider_fatal"
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

// recordSynthesis stores a completed text-to-speech synthesis for the current customer.
func (s *APIIntegrationSuite) recordSynthesis(at time.Time, characters, costMicros int64) {
	latencyMs := 180.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &store.Request{
		Modality:   "tts",
		CustomerID: s.customerID,
		Provider:   "elevenlabs",
		Model:      "eleven_flash_v2_5",
		StartedAt:  at,
		AudioMs:    2500,
		Characters: characters,
		CostMicros: costMicros,
		LatencyMs:  &latencyMs,
		Success:    true,
	}))
}

func (s *APIIntegrationSuite) TestHealthReportsBothDependenciesAsOk() {
	response, payload := s.do(http.MethodGet, "/health", "")

	s.Equal(http.StatusOK, response.StatusCode)

	var status HealthStatus
	s.Require().NoError(json.Unmarshal(payload, &status))
	s.Equal(Ok, status.Status)
	s.Equal("ok", status.Dependencies["postgres"])
	s.Equal("ok", status.Dependencies["redis"])
	s.Equal("ok", status.Dependencies["stt"])
	s.Equal("ok", status.Dependencies["tts"])
}

func (s *APIIntegrationSuite) TestRollupThenStatsReportsTheCustomersUsage() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 120, true)
	s.recordTurn(s.base.Add(6*time.Minute), 2000, 240, true)
	s.recordTurn(s.base.Add(7*time.Minute), 1000, 360, false)

	window := fmt.Sprintf(
		`{"granularity":"hourly","from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var rollup RollupResult
	s.Require().NoError(json.Unmarshal(payload, &rollup))
	s.Equal(Hourly, rollup.Granularity)
	s.Positive(rollup.BucketsWritten)

	path := fmt.Sprintf(
		"/v1/stt/stats?granularity=hourly&from=%s&to=%s",
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload = s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1)

	bucket := buckets[0]
	s.Equal("deepgram", bucket.Provider)
	s.Equal("flux-general-en", bucket.Model)
	s.EqualValues(6000, bucket.AudioMsTotal, "audio duration is what providers bill for")
	s.EqualValues(3, bucket.RequestCount)
	s.EqualValues(1, bucket.ErrorCount)
	s.Require().NotNil(bucket.LatencyP50Ms)
	s.InDelta(240.0, *bucket.LatencyP50Ms, 0.001)
	s.Require().NotNil(bucket.Uptime)
	s.InDelta(2.0/3.0, *bucket.Uptime, 0.001)
}

func (s *APIIntegrationSuite) TestStatsAreReportedPerModality() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)
	s.recordSynthesis(s.base.Add(6*time.Minute), 128, 6400)

	window := fmt.Sprintf(
		`{"from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, _ := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	from, to := s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339)

	response, payload := s.do(http.MethodGet, fmt.Sprintf("/v1/stt/stats?from=%s&to=%s", from, to), "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var transcription []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &transcription))
	s.Require().Len(transcription, 1, "the synthesis belongs to the other modality")
	s.EqualValues(3000, transcription[0].AudioMsTotal)

	response, payload = s.do(http.MethodGet, fmt.Sprintf("/v1/tts/stats?from=%s&to=%s", from, to), "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var synthesis []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &synthesis))
	s.Require().Len(synthesis, 1)
	s.Equal("elevenlabs", synthesis[0].Provider)
	s.EqualValues(128, synthesis[0].CharactersTotal)
	s.EqualValues(6400, synthesis[0].CostMicrosTotal, "cost is aggregated alongside usage")
}

func (s *APIIntegrationSuite) TestStatsAreScopedToTheCallingCustomer() {
	s.recordTurn(s.base.Add(5*time.Minute), 3000, 100, true)

	// Another customer's traffic in the same bucket must not show up.
	other := &store.Request{
		Modality:   "stt",
		CustomerID: s.customerID + "-other",
		Provider:   "deepgram",
		Model:      "flux-general-en",
		StartedAt:  s.base.Add(5 * time.Minute),
		AudioMs:    99000,
		Success:    true,
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, other))

	window := fmt.Sprintf(
		`{"from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, _ := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode)

	path := fmt.Sprintf(
		"/v1/stt/stats?from=%s&to=%s",
		s.base.Format(time.RFC3339), s.base.Add(time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1)
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *APIIntegrationSuite) TestDailyGranularityCollapsesTheHours() {
	s.recordTurn(s.base.Add(1*time.Hour), 1000, 100, true)
	s.recordTurn(s.base.Add(6*time.Hour), 2000, 100, true)

	window := fmt.Sprintf(
		`{"granularity":"daily","from":%q,"to":%q}`,
		s.base.Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339),
	)
	response, payload := s.do(http.MethodPost, "/v1/stats/rollup", window)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var rollup RollupResult
	s.Require().NoError(json.Unmarshal(payload, &rollup))
	s.Equal(Daily, rollup.Granularity)

	path := fmt.Sprintf(
		"/v1/stt/stats?granularity=daily&from=%s&to=%s",
		s.base.Add(-24*time.Hour).Format(time.RFC3339), s.base.Add(24*time.Hour).Format(time.RFC3339),
	)
	response, payload = s.do(http.MethodGet, path, "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var buckets []StatsBucket
	s.Require().NoError(json.Unmarshal(payload, &buckets))
	s.Require().Len(buckets, 1, "both hours belong to the same day")
	s.EqualValues(3000, buckets[0].AudioMsTotal)
}

func (s *APIIntegrationSuite) TestProvidersReportLiveHealth() {
	s.Require().NoError(s.live.RecordRequest(s.ctx, live.Usage{
		Modality: "stt", CustomerID: s.customerID,
		Provider: "deepgram", Model: "flux-general-en",
		LatencyMs: 150, AudioMs: 1000, Success: true,
	}))

	response, payload := s.do(http.MethodGet, "/v1/stt/providers", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var providers []Provider
	s.Require().NoError(json.Unmarshal(payload, &providers))

	var english *Provider
	for i := range providers {
		if providers[i].Model == "flux-general-en" {
			english = &providers[i]
		}
	}
	s.Require().NotNil(english)
	s.Positive(english.Health.Requests, "health should come from the live counters")
	s.True(english.Health.Available)
}

func (s *APIIntegrationSuite) TestVoiceProvidersAreRankedSeparately() {
	// A speech-to-text failure must not make the text-to-speech provider look unhealthy.
	s.Require().NoError(s.live.RecordRequest(s.ctx, live.Usage{
		Modality: "stt", CustomerID: s.customerID,
		Provider: "elevenlabs", Model: "eleven_flash_v2_5",
		LatencyMs: 150, Success: false,
	}))

	response, payload := s.do(http.MethodGet, "/v1/tts/providers", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var providers []Provider
	s.Require().NoError(json.Unmarshal(payload, &providers))
	s.Require().NotEmpty(providers)

	for _, provider := range providers {
		if provider.Model == "eleven_flash_v2_5" {
			s.Zero(provider.Health.Errors, "health is keyed by modality")
		}
	}
}
