//go:build integration

package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
)

// haiku is the model the regions suite runs in the us and in the eu, the one the fixtures'
// price file prices in both.
const haiku = anthropic.ProviderName + "/claude-haiku-5-5"

// The variables the suite's two Claude instances name their keys in.
const (
	regionsUSKey = "REGIONS_SUITE_ANTHROPIC_US_KEY"
	regionsEUKey = "REGIONS_SUITE_ANTHROPIC_EU_KEY"
)

// RegionsSuite runs the router with Claude Haiku declared once in the us and once in the eu,
// each instance at an endpoint of its own on this machine with a key of its own, the scheduling
// model in the eu as well, a turbopuffer in the eu beside the us knowledge base, and the eu's
// sandboxes on a Daytona target of their own. Rows are priced from the database, seeded with
// the fixtures' price file, which prices the eu. An agent's region is what its sync or its
// config names.
type RegionsSuite struct {
	RouterSuite
	us, eu  *claudeHost
	euKnows *turbopuffertest.Server
}

func TestRegionsSuite(t *testing.T) {
	runSuite(t, new(RegionsSuite))
}

func (s *RegionsSuite) SetupSuite() {
	s.us, s.eu = s.claudeAt("from the us"), s.claudeAt("from the eu")
	// The suites share the process, so each variable is the suite's own.
	s.Require().NoError(os.Setenv(regionsUSKey, "us-key"))
	s.Require().NoError(os.Setenv(regionsEUKey, "eu-key"))
	model := strings.TrimPrefix(haiku, anthropic.ProviderName+"/")
	s.llmInstances = []routing.ProviderConfig{
		{Provider: anthropic.ProviderName, Model: model, Languages: []string{"la"}, Region: pricing.RegionUS,
			Endpoint: s.us.URL, APIKeyEnv: regionsUSKey},
		{Provider: anthropic.ProviderName, Model: model, Languages: []string{"la"}, Region: pricing.RegionEU,
			Endpoint: s.eu.URL, APIKeyEnv: regionsEUKey},
		{Provider: "scheduling", Model: "scheduling-model", Languages: []string{"la"}, Region: pricing.RegionEU},
		// What the flow controller and the title of an eu agent's session run on (llm-flow, llm-fast).
		{Provider: "stub", Model: "stub-model", Languages: []string{"en"}, Realtime: true, Region: pricing.RegionEU},
	}
	vendors := llmrouter.DefaultRegistry()
	s.llmFactories = func(registry *llmrouter.Registry) {
		registry.Register(anthropic.ProviderName, func(spec routing.Spec) (llmrouter.Provider, error) {
			return vendors.Build(anthropic.ProviderName, spec)
		}, routing.OverrideAPIKey, routing.OverrideBaseURL)
		// The stubs run in process, so they are wherever they are asked to be.
		registry.Register("scheduling", s.schedulingModel, routing.OverrideRegion)
		registry.Register("stub", func(routing.Spec) (llmrouter.Provider, error) { return &scriptedLLM{}, nil }, routing.OverrideRegion)
	}
	s.euKnows = turbopuffertest.Start(s.T())
	euKnowledge, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: s.euKnows.URL})
	s.Require().NoError(err)
	s.knowledgeRegions = map[string]knowledge.Backend{pricing.RegionEU: euKnowledge}
	s.daytona = daytonatest.Start(s.T())
	s.sandboxTargets = map[string]string{pricing.RegionEU: "eu"}
	s.pricedFromDatabase = true
	s.RouterSuite.SetupSuite()
}

func (s *RegionsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// claudeHost is one instance's Anthropic API on this machine: it answers every request with
// its words, and keeps the key each was sent with.
type claudeHost struct {
	*httptest.Server
	mu   sync.Mutex
	keys []string
}

func (s *RegionsSuite) claudeAt(words string) *claudeHost {
	host := &claudeHost{}
	host.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host.mu.Lock()
		host.keys = append(host.keys, r.Header.Get("X-Api-Key"))
		host.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, claudeStream(words))
	}))
	s.T().Cleanup(host.Close)
	return host
}

// asked is how many requests the host has had.
func (h *claudeHost) asked() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.keys)
}

func (h *claudeHost) lastKey() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.keys[len(h.keys)-1]
}

// claudeStream is a Messages API answer of words, 12 tokens read and 3 written.
func claudeStream(words string) string {
	quoted, _ := json.Marshal(words)
	var stream strings.Builder
	for _, event := range []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-5-5","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":` + string(quoted) + `}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	} {
		var typed struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &typed)
		fmt.Fprintf(&stream, "event: %s\ndata: %s\n\n", typed.Type, event)
	}
	return stream.String()
}

func (s *RegionsSuite) TestADirectCallAskingForARegionNothingRunsInIsRefusedNamingIt() {
	eu := pricing.RegionEU

	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings", SpeechRequest{
		Text: "hello", Inline: pointerTo(true), Options: &TtsOptions{DataPolicy: &DataPolicy{Region: &eu}},
	}, &spoken))

	s.Equal(RecordingStatusFailed, spoken.Status)
	s.Require().NotNil(spoken.Error)
	s.Contains(*spoken.Error, "no candidate in region eu")
}

func (s *RegionsSuite) TestAPolicyCanNoLongerCarryARegion() {
	status, _ := s.serverClient.failure(http.MethodPut, "/v1/policies/app",
		map[string]any{"data_policy": map[string]any{"region": pricing.RegionEU}})

	s.Equal(http.StatusBadRequest, status)
	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))
	s.Nil(read.DataPolicy, "nothing was stored")
}

func (s *RegionsSuite) TestAPolicysDataPolicyStillHoldsTrainingAndRetention() {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/app",
		Policy{DataPolicy: &PolicyDataPolicy{AllowTraining: pointerTo(false), Retention: pointerTo("30d")}}, nil))

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))

	s.Require().NotNil(read.DataPolicy)
	s.False(value(read.DataPolicy.AllowTraining))
	s.Equal("30d", value(read.DataPolicy.Retention))
}

func (s *RegionsSuite) TestTheProvidersListNamesEachInstancesRegion() {
	var listed []Provider
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/llm/providers", nil, &listed))

	var regions []string
	for _, provider := range listed {
		if provider.Provider+"/"+provider.Model == haiku {
			regions = append(regions, provider.Region)
		}
	}
	s.Equal([]string{pricing.RegionUS, pricing.RegionEU}, regions)
}
