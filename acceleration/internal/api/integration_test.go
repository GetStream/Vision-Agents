//go:build integration

package api

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
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
		Store:  pgStore,
		Live:   liveClient,
		Voices: s.voiceService(pgStore),
		// Minting a token signs one rather than fetching it, so a made-up app is enough
		// to exercise the join path without a real Stream account behind it.
		StreamKey:    testStreamKey,
		StreamSecret: testStreamSecret,
	})
	s.Require().NoError(err)
	s.server = httptest.NewServer(server.Handler())

	s.base = time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
}

// voiceService wires the voice paths against a directory and a provider that always takes
// the recordings, so the HTTP surface can be exercised without cloning anything for real.
func (s *APIIntegrationSuite) voiceService(pgStore *store.Store) *voices.Service {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"voice_id":"el-cloned"}`))
	}))
	s.T().Cleanup(provider.Close)

	bucket, err := blob.Open(s.ctx, "file://"+s.T().TempDir())
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(bucket.Close()) })

	cloner, err := voices.NewElevenLabs(voices.ElevenLabsOptions{APIKey: "secret", BaseURL: provider.URL})
	s.Require().NoError(err)
	cloners := voices.NewRegistry()
	cloners.Register("elevenlabs", cloner)

	service, err := voices.NewService(voices.Options{Store: pgStore, Bucket: bucket, Cloners: cloners})
	s.Require().NoError(err)
	return service
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

func (s *APIIntegrationSuite) TestAnAgentConfigSurvivesBeingStoredAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/agents/configs", `{
		"name":"support","llm":"llm-fast","tts":"en-low-latency","voice":"aurora",
		"subagent":"llm-best","instructions":"be brief","skills":["think","refund"],
		"keyterms":["Vision Agents","Stream"],
		"knowledge_namespace":"handbook","tags":{"project":"support"}
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotEmpty(created.Id)
	s.Equal("support", created.Name)

	response, payload = s.do(http.MethodGet, "/v1/agents/configs/"+created.Id, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var read AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Require().NotNil(read.Llm)
	s.Equal("llm-fast", *read.Llm)
	s.Require().NotNil(read.Voice)
	s.Equal("aurora", *read.Voice)
	s.Require().NotNil(read.Skills)
	s.Equal([]string{"think", "refund"}, *read.Skills)
	s.Require().NotNil(read.Keyterms)
	s.Equal([]string{"Vision Agents", "Stream"}, *read.Keyterms)
	s.Require().NotNil(read.KnowledgeNamespace)
	s.Equal("handbook", *read.KnowledgeNamespace)
	s.Require().NotNil(read.Tags)
	s.Equal("support", (*read.Tags)["project"])
}

func (s *APIIntegrationSuite) TestAConfigNamingMoreKeytermsThanAnyProviderTakesIsRefused() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = fmt.Sprintf("%q", fmt.Sprintf("term-%d", i))
	}
	body := fmt.Sprintf(`{"name":"support","keyterms":[%s]}`, strings.Join(terms, ","))

	response, payload := s.do(http.MethodPost, "/v1/agents/configs", body)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "keyterms")
}

func (s *APIIntegrationSuite) TestUpdatingAConfigReplacesWhatItWas() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs",
		`{"name":"support","llm":"llm-fast","instructions":"be brief"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		`{"name":"support","llm":"llm-best"}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var updated AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &updated))
	s.Equal(created.Id, updated.Id, "an update keeps the id callers already hold")
	s.Require().NotNil(updated.Llm)
	s.Equal("llm-best", *updated.Llm)
	s.Nil(updated.Instructions, "a field left out of a replacement is gone from it")
}

func (s *APIIntegrationSuite) TestADeletedConfigCannotBeUsedAgain() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, _ := s.do(http.MethodDelete, "/v1/agents/configs/"+created.Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	response, _ = s.do(http.MethodGet, "/v1/agents/configs/"+created.Id, "")
	s.Equal(http.StatusNotFound, response.StatusCode)

	// The name is free again, which is what makes deleting one usable rather than final.
	response, payload = s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	s.Equal(http.StatusCreated, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestAnotherCustomersConfigIsNotFound() {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"support"}`)
	var created AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &created))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/agents/configs/"+created.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestASkillIsStoredAndListed() {
	response, payload := s.do(http.MethodPost, "/v1/agents/skills", `{
		"name":"refund","description":"work out what a caller is owed",
		"instructions":"Read the order and the policy, then say what to refund.",
		"deadline_ms":20000
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Skill
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotNil(created.DeadlineMs)
	s.EqualValues(20000, *created.DeadlineMs)

	response, payload = s.do(http.MethodGet, "/v1/agents/skills", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var listed []Skill
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal("refund", listed[0].Name)
}

func (s *APIIntegrationSuite) TestASkillWithoutADescriptionIsRefused() {
	// The description is the whole of how the fast model decides when to hand work over,
	// so a skill without one would never be reached for.
	response, payload := s.do(http.MethodPost, "/v1/agents/skills",
		`{"name":"refund","description":"","instructions":"work it out"}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "description")
}

func (s *APIIntegrationSuite) TestSyncingAnAgentStoresItsInstructionsAndSkills() {
	response, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v1",
		"instructions":"Be brief.",
		"skills":[{"name":"refund","description":"work out a refund","instructions":"Read the policy."}]
	}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var first SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &first))
	s.False(first.Unchanged)
	s.Equal("support", first.Config.Name)
	s.Require().NotNil(first.Config.Instructions)
	s.Equal("Be brief.", *first.Config.Instructions)
	s.Require().NotNil(first.Config.Skills)
	s.Equal([]string{"refund"}, *first.Config.Skills)
	s.Require().NotNil(first.Config.SyncHash)
	s.Equal("v1", *first.Config.SyncHash)

	again, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v1",
		"instructions":"Be brief.",
		"skills":[{"name":"refund","description":"work out a refund","instructions":"Read the policy."}]
	}`)
	s.Require().Equal(http.StatusOK, again.StatusCode, string(payload))

	var second SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &second))
	s.True(second.Unchanged, "the same hash means nothing was written")
	s.Equal(first.Config.Id, second.Config.Id)

	changed, payload := s.do(http.MethodPost, "/v1/agents/sync", `{
		"name":"support","hash":"v2",
		"instructions":"Be even briefer."
	}`)
	s.Require().Equal(http.StatusOK, changed.StatusCode, string(payload))

	var third SyncAgentResult
	s.Require().NoError(json.Unmarshal(payload, &third))
	s.False(third.Unchanged)
	s.Equal(first.Config.Id, third.Config.Id)
	s.Require().NotNil(third.Config.Instructions)
	s.Equal("Be even briefer.", *third.Config.Instructions)
}

// campaign creates a campaign over a stored config and returns it.
func (s *APIIntegrationSuite) campaign(concurrency int) Campaign {
	_, payload := s.do(http.MethodPost, "/v1/agents/configs", `{"name":"winback","llm":"llm-fast"}`)
	var config AgentConfig
	s.Require().NoError(json.Unmarshal(payload, &config))

	body := fmt.Sprintf(
		`{"name":"may","config_id":%q,"from_number":"+15550100","concurrency":%d}`,
		config.Id, concurrency)
	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns", body)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Campaign
	s.Require().NoError(json.Unmarshal(payload, &created))
	return created
}

func (s *APIIntegrationSuite) TestACampaignIsCreatedStoppedWithNobodyToRing() {
	created := s.campaign(3)

	s.Equal(Draft, created.State, "a campaign that started itself would ring people nobody added yet")
	s.Equal(3, created.Concurrency)

	response, payload := s.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id+"/contacts", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var contacts []Contact
	s.Require().NoError(json.Unmarshal(payload, &contacts))
	s.Empty(contacts)
}

func (s *APIIntegrationSuite) TestACampaignNamingAConfigNobodyHasIsRefused() {
	// A campaign that named a config it could not use would fail one call at a time, at
	// whatever hour somebody started it.
	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns",
		`{"name":"may","config_id":"nope","from_number":"+15550100"}`)

	s.Require().Equal(http.StatusBadRequest, response.StatusCode)

	var failure Error
	s.Require().NoError(json.Unmarshal(payload, &failure))
	s.Contains(failure.Error, "config")
}

func (s *APIIntegrationSuite) TestContactsAreRungInTheOrderTheyWereAdded() {
	created := s.campaign(1)

	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts", `{
		"contacts":[
			{"to_number":"+15550111","instructions":"ask about the trial"},
			{"to_number":"+15550222"}
		]
	}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	claimed, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal("+15550111", claimed.ToNumber)
	s.Equal("ask about the trial", claimed.Instructions)
	s.Equal(store.Calling, claimed.State, "a claimed contact is nobody else's to ring")
	s.Equal(1, claimed.Attempts)

	next, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal("+15550222", next.ToNumber, "the same person was taken twice")

	_, found, err = s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.False(found, "there was nobody left to ring")
}

func (s *APIIntegrationSuite) TestACallThatNeverFinishedIsRungAgainRatherThanLost() {
	// A process that stopped mid-call leaves a contact claimed by nobody. Ringing them
	// again is better than a campaign that quietly skips them.
	created := s.campaign(1)
	_, _ = s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts",
		`{"contacts":[{"to_number":"+15550111"}]}`)

	_, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)

	s.Require().NoError(s.store.ReleaseContacts(s.ctx, created.Id))

	again, found, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().True(found)
	s.Equal(2, again.Attempts, "the second attempt is counted as one")
}

func (s *APIIntegrationSuite) TestWhatBecameOfAContactIsShownAgainstIt() {
	created := s.campaign(1)
	_, _ = s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/contacts",
		`{"contacts":[{"to_number":"+15550111"},{"to_number":"+15550222"}]}`)

	first, _, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(s.ctx, store.Contact{
		ID: first.ID, State: store.Done, CallID: "session-1", VendorCallID: "CA1",
	}))

	second, _, err := s.store.ClaimContact(s.ctx, created.Id)
	s.Require().NoError(err)
	s.Require().NoError(s.store.FinishContact(s.ctx, store.Contact{
		ID: second.ID, State: store.Failed, Error: "the number is not in service",
	}))

	response, payload := s.do(http.MethodGet, "/v1/agents/campaigns/"+created.Id+"/contacts", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var contacts []Contact
	s.Require().NoError(json.Unmarshal(payload, &contacts))
	s.Require().Len(contacts, 2)
	s.Equal(ContactStateDone, contacts[0].State)
	s.Require().NotNil(contacts[0].CallId)
	s.Equal("session-1", *contacts[0].CallId)
	s.Equal(ContactStateFailed, contacts[1].State)
	s.Require().NotNil(contacts[1].Error)
	s.Contains(*contacts[1].Error, "not in service")
}

func (s *APIIntegrationSuite) TestACampaignCannotBeStartedWithoutAnythingToRingWith() {
	// This deployment has no telephony, so starting one has to say so rather than
	// reporting a campaign that is running and will never call anybody.
	created := s.campaign(1)

	response, payload := s.do(http.MethodPost, "/v1/agents/campaigns/"+created.Id+"/start", "")

	s.Require().Equal(http.StatusBadRequest, response.StatusCode, string(payload))
}

func (s *APIIntegrationSuite) TestAnotherCustomersCampaignIsNotFound() {
	created := s.campaign(1)

	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet,
		s.server.URL+"/v1/agents/campaigns/"+created.Id, strings.NewReader(""))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestACallIsFoundAfterTheSessionRunningItIsGone() {
	// A session lives in a map in memory. This is the whole point of the row: the call
	// is still there once the process that held it is not.
	call := store.Call{
		ID:         "session-" + s.customerID,
		CustomerID: s.customerID,
		CallID:     "call-1",
		AgentID:    "agent-1",
		Direction:  store.Outbound,
		ToNumber:   "+15550101",
		StartedAt:  s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var read Call
	s.Require().NoError(json.Unmarshal(payload, &read))
	s.Equal("call-1", read.CallId)
	s.Equal(Outbound, read.Direction)
	s.Require().NotNil(read.ToNumber)
	s.Equal("+15550101", *read.ToNumber)
	s.Nil(read.EndedAt, "a call nobody has ended is still running")
}

// testStreamKey and testStreamSecret stand in for a Stream app. The secret signs the join
// tokens, which is what lets a test read one back and see who it was minted for.
const (
	testStreamKey    = "test-app"
	testStreamSecret = "test-secret"
)

func (s *APIIntegrationSuite) TestAJoinTokenSaysWhichCallToJoinAndWhoAsIt() {
	// The browser is handed a token and a call, never the secret: whoever holds this can
	// join one call as one user until it expires, and can sign nothing of their own.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, payload := s.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token", `{}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var minted CallToken
	s.Require().NoError(json.Unmarshal(payload, &minted))
	s.Equal(testStreamKey, minted.ApiKey)
	s.Equal("call-1", minted.CallId, "the Stream call, not the id we hold it by")
	s.Equal("default", minted.CallType)
	s.NotEqual("agent-1", minted.UserId, "a listener is not the agent")
	s.True(minted.ExpiresAt.After(time.Now()), "a token that has expired is no use")

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(testStreamSecret), nil
	})
	s.Require().NoError(err, "the token is signed with the app secret")
	s.Equal(minted.UserId, claimed["user_id"], "and it is signed for the user it names")
}

func (s *APIIntegrationSuite) TestAJoinTokenIsMintedForTheUserTheCallerAsksFor() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	_, payload := s.do(http.MethodPost, "/v1/agents/calls/"+call.ID+"/token",
		`{"user_id":"thierry","user_name":"Thierry"}`)

	var minted CallToken
	s.Require().NoError(json.Unmarshal(payload, &minted))
	s.Equal("thierry", minted.UserId)
	s.Equal("Thierry", minted.UserName)

	claimed := jwt.MapClaims{}
	_, err := jwt.ParseWithClaims(minted.Token, claimed, func(*jwt.Token) (any, error) {
		return []byte(testStreamSecret), nil
	})
	s.Require().NoError(err)
	s.Equal("thierry", claimed["user_id"])
}

func (s *APIIntegrationSuite) TestACallAnotherCustomerHoldsCannotBeJoined() {
	// Handing out a token for somebody else's call would be handing out their call.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost,
		s.server.URL+"/v1/agents/calls/"+call.ID+"/token", strings.NewReader(`{}`))
	s.Require().NoError(err)
	request.Header.Set(CustomerHeader, "somebody-else")

	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestTheRunningCallsAreTheOnesThatHaveNotEnded() {
	running := store.Call{
		ID: "running-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	finished := store.Call{
		ID: "finished-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-2", AgentID: "agent-2", StartedAt: s.base.Add(-time.Hour),
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &running))
	s.Require().NoError(s.store.StartCall(s.ctx, &finished))
	s.Require().NoError(s.store.FinishCall(s.ctx, finished.ID, s.base))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls?running=true", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var listed []Call
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 1)
	s.Equal(running.ID, listed[0].Id)

	response, payload = s.do(http.MethodGet, "/v1/agents/calls", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Require().Len(listed, 2, "both calls happened")
	s.Equal(running.ID, listed[0].Id, "newest first")
	s.Require().NotNil(listed[1].EndedAt)
}

func (s *APIIntegrationSuite) TestACallKeepsTheTimeItFirstEnded() {
	// An agent leaves once. A second close is the same leaving reported again, and must
	// not stretch the call to cover it.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Minute)))
	s.Require().NoError(s.store.FinishCall(s.ctx, call.ID, s.base.Add(time.Hour)))

	read, err := s.store.Call(s.ctx, s.customerID, call.ID)
	s.Require().NoError(err)
	s.Require().NotNil(read.EndedAt)
	s.WithinDuration(s.base.Add(time.Minute), *read.EndedAt, time.Second)
}

func (s *APIIntegrationSuite) TestWhatACallDecidedIsFoundByTheRowRecordingIt() {
	// A dashboard holds the row's id, and the agent wrote its reasoning against the call
	// it joined. Reading one by the other is what puts the decision log on the page.
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: s.customerID,
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))
	s.Require().NoError(s.store.RecordCallEvents(s.ctx, []store.CallEvent{{
		CustomerID: s.customerID, CallID: call.CallID, AgentID: call.AgentID,
		At: s.base.Add(time.Second), Kind: "answer", Reason: "a complete thought",
		Said: "how is the weather",
	}}))

	response, payload := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID+"/events", "")
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var decided []CallEvent
	s.Require().NoError(json.Unmarshal(payload, &decided))
	s.Require().Len(decided, 1)
	s.Equal(DecisionKind("answer"), decided[0].Kind)
	s.Require().NotNil(decided[0].Said)
	s.Equal("how is the weather", *decided[0].Said)
}

func (s *APIIntegrationSuite) TestAnotherCustomersCallIsNotFound() {
	call := store.Call{
		ID: "session-" + s.customerID, CustomerID: "somebody-else",
		CallID: "call-1", AgentID: "agent-1", StartedAt: s.base,
	}
	s.Require().NoError(s.store.StartCall(s.ctx, &call))

	response, _ := s.do(http.MethodGet, "/v1/agents/calls/"+call.ID, "")
	s.Equal(http.StatusNotFound, response.StatusCode)
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

func (s *APIIntegrationSuite) TestAVoiceIsRecordedPreparedAndReadBack() {
	response, payload := s.do(http.MethodPost, "/v1/agents/voices",
		`{"name":"founder","description":"the one from the ad"}`)
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))
	s.Require().NotEmpty(created.Id)
	s.Equal("founder", created.Name)
	s.Empty(*created.Samples, "a voice starts with nothing recorded")

	audio := base64.StdEncoding.EncodeToString([]byte("pretend this is speech"))
	response, payload = s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/samples",
		fmt.Sprintf(`{"audio":%q,"filename":"clip.wav","content_type":"audio/wav","transcript":"hello"}`, audio))
	s.Require().Equal(http.StatusCreated, response.StatusCode, string(payload))

	var recorded Voice
	s.Require().NoError(json.Unmarshal(payload, &recorded))
	s.Require().Len(*recorded.Samples, 1)
	s.EqualValues(22, *(*recorded.Samples)[0].Bytes)

	response, payload = s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/prepare", `{}`)
	s.Require().Equal(http.StatusOK, response.StatusCode, string(payload))

	var prepared Voice
	s.Require().NoError(json.Unmarshal(payload, &prepared))
	s.Require().Len(*prepared.Bindings, 1)
	s.Equal(VoiceBindingStateReady, (*prepared.Bindings)[0].State)
	s.Equal("el-cloned", *(*prepared.Bindings)[0].ExternalId,
		"a session names the voice, and the provider is asked for its own id")
}

func (s *APIIntegrationSuite) TestAVoiceWithNothingRecordedCannotBePrepared() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, payload := s.do(http.MethodPost, "/v1/agents/voices/"+created.Id+"/prepare", `{}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "add a recording")
}

func (s *APIIntegrationSuite) TestAVoiceBelongingToSomebodyElseIsNotThere() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	s.customerID = "somebody-else"
	response, _ := s.do(http.MethodGet, "/v1/agents/voices/"+created.Id, "")

	s.Equal(http.StatusNotFound, response.StatusCode)
}

func (s *APIIntegrationSuite) TestAVoiceNeedsAName() {
	response, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"  "}`)

	s.Equal(http.StatusBadRequest, response.StatusCode, string(payload))
	s.Contains(string(payload), "needs a name")
}

func (s *APIIntegrationSuite) TestADeletedVoiceStopsBeingListed() {
	_, payload := s.do(http.MethodPost, "/v1/agents/voices", `{"name":"founder"}`)
	var created Voice
	s.Require().NoError(json.Unmarshal(payload, &created))

	response, _ := s.do(http.MethodDelete, "/v1/agents/voices/"+created.Id, "")
	s.Require().Equal(http.StatusNoContent, response.StatusCode)

	response, payload = s.do(http.MethodGet, "/v1/agents/voices", "")
	s.Require().Equal(http.StatusOK, response.StatusCode)

	var listed []Voice
	s.Require().NoError(json.Unmarshal(payload, &listed))
	s.Empty(listed)
}
