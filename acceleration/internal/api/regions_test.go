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
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer/turbopuffertest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/anthropic"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing/pricingtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
// the fixtures' price file, which prices the eu.
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
		// What the flow controller of an eu app's session runs on (llm-flow).
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

// keepIn sets the scope's policy to keep its work in region.
func (s *RegionsSuite) keepIn(scope, region string) {
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/policies/"+scope,
		Policy{DataPolicy: &DataPolicy{Region: &region}}, nil))
}

// runOn runs model once and answers the status and what came back.
func (s *RegionsSuite) runOn(model string) (int, Run, string) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/runs", CreateRunRequest{ConfigId: s.config(model), Input: "hello"})
	var run Run
	_ = json.Unmarshal(payload, &run)
	var refused ErrorResponse
	_ = json.Unmarshal(payload, &refused)
	return status, run, refused.Error.Message
}

// rowsOf are the app's rows of provider, once at least one is written.
func (s *RegionsSuite) rowsOf(provider string) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		err := s.store.ShardDB().NewSelect().Model(&rows).
			Where("customer_id = ? AND provider = ?", s.customerID(), provider).Order("started_at").Scan(context.Background())
		return err == nil && len(rows) > 0
	}, settleFor, 50*time.Millisecond, "no %s row was written", provider)
	return rows
}

// pricedAt is what a row's work costs at item's entry of the fixtures' price file.
func (s *RegionsSuite) pricedAt(item pricing.Item, row store.Request) int64 {
	return pricingtest.Micros(s.T(), pricingtest.List(s.T(), "fixtures"), item, row.Quantities())
}

func (s *RegionsSuite) TestAnOrganizationInTheEUIsServedOnlyByTheEUInstanceAndBilledAtItsEntry() {
	s.keepIn("organization", pricing.RegionEU)
	usAsked := s.us.asked()

	status, run, refusal := s.runOn(haiku)

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal("from the eu", run.FinalText)
	s.Equal("eu-key", s.eu.lastKey(), "the eu instance's own key")
	s.Equal(usAsked, s.us.asked(), "the us instance was never asked")
	row := s.rowsOf(anthropic.ProviderName)[0]
	s.Equal("llm-anthropic-claude-haiku-5-5-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.LLM(pricing.RegionEU, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestAnAppInTheEUWhoseOrganizationNamesNoRegionIsServedInTheEU() {
	s.keepIn("app", pricing.RegionEU)

	status, run, refusal := s.runOn(haiku)

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal("from the eu", run.FinalText)
	s.Equal("llm-anthropic-claude-haiku-5-5-eu", s.rowsOf(anthropic.ProviderName)[0].SKU)
}

func (s *RegionsSuite) TestWorkNobodyAskedARegionForIsServedInTheUS() {
	status, run, refusal := s.runOn(haiku)

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal("from the us", run.FinalText)
	s.Equal("us-key", s.us.lastKey())
	row := s.rowsOf(anthropic.ProviderName)[0]
	s.Equal("llm-anthropic-claude-haiku-5-5-us", row.SKU)
	s.Equal(s.pricedAt(pricingtest.LLM(pricing.RegionUS, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestAModelWithNoEUInstanceIsRefusedNamingTheRegion() {
	s.keepIn("organization", pricing.RegionEU)

	status, _, refusal := s.runOn("counted/counted-model")

	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "no candidate in region eu")
}

func (s *RegionsSuite) TestAnOrganizationInTheEUAndAnAppInTheUSAreRefusedNamingBoth() {
	s.keepIn("organization", pricing.RegionEU)
	s.keepIn("app", pricing.RegionUS)

	status, _, refusal := s.runOn(haiku)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "keeps its work in eu and the app's in us")
}

func (s *RegionsSuite) TestARequestAskingForTheUSUnderAnOrganizationInTheEUIsRefusedNamingBoth() {
	s.keepIn("organization", pricing.RegionEU)
	us := pricing.RegionUS

	var spoken Speech
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/tts/recordings", SpeechRequest{
		Text: "hello", Inline: pointerTo(true), Options: &TtsOptions{DataPolicy: &DataPolicy{Region: &us}},
	}, &spoken))

	s.Equal(RecordingStatusFailed, spoken.Status)
	s.Require().NotNil(spoken.Error)
	s.Contains(*spoken.Error, "the data policies name different regions: us and eu")
}

func (s *RegionsSuite) TestARegionOfAnotherFormIsRefusedInAPolicy() {
	status, _ := s.serverClient.failure(http.MethodPut, "/v1/policies/app", Policy{DataPolicy: &DataPolicy{Region: pointerTo("EU West")}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *RegionsSuite) TestAPolicysRegionReadsBack() {
	s.keepIn("app", pricing.RegionEU)

	var read Policy
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/policies/app", nil, &read))

	s.Require().NotNil(read.DataPolicy)
	s.Equal(pricing.RegionEU, value(read.DataPolicy.Region))
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

func (s *RegionsSuite) TestAnEUAppsKnowledgeIsWrittenInTheEUAndBilledAtItsEntry() {
	s.keepIn("app", pricing.RegionEU)
	namespace := "docs-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/knowledge", IngestKnowledgeRequest{
		Namespace: namespace, Documents: []KnowledgeDocument{{Source: "refunds.md", Text: "# Refunds\n\nRefunds take thirty days."}},
	}, nil))

	scoped := knowledge.Scoped(s.customerID(), namespace)
	written, _ := s.knowledge.stored()
	s.NotEqual(scoped, written, "nothing of an eu app's is written in the us")
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: s.euKnows.URL})
	s.Require().NoError(err)
	names, err := backend.Namespaces(context.Background(), scoped)
	s.Require().NoError(err)
	s.Equal([]string{scoped}, names)
	row := s.rowsOf("turbopuffer")[0]
	s.Equal("knowledge-turbopuffer-write-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.Knowledge(pricing.RegionEU, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestAnEUAppsSandboxIsMadeOnTheEUTargetAndBilledAtItsEntry() {
	s.keepIn("app", pricing.RegionEU)
	made := len(s.daytona.Creations())
	config := s.sandboxedConfig()
	s.codeHandsBackAReport()

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/runs", CreateRunRequest{ConfigId: config, Input: "how did I walk?"})

	s.Require().Equal(http.StatusOK, status, string(payload))
	creations := s.daytona.Creations()
	s.Require().Len(creations, made+1)
	s.Equal("eu", creations[made]["target"], "made on the eu's Daytona target")
	row := s.rowsOf("daytona")[0]
	s.Equal("sandbox-daytona-running-class-container-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.Sandbox(pricing.RegionEU, false), row), row.CostMicros)
}

// sandboxedConfig is an agent whose model runs the code its test queues, in a sandbox.
func (s *RegionsSuite) sandboxedConfig() string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("scheduling/scheduling-model"), Sandbox: pointerTo(Daytona),
	}, &created))
	return created.Id
}
