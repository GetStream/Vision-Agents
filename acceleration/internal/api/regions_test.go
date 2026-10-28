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

// synced is a sync of an agent directory declaring the settings given, under a name of the
// test's own unless they name one: the status, the config stored and the refusal.
func (s *RegionsSuite) synced(declared map[string]any) (int, AgentConfig, string) {
	if _, named := declared["name"]; !named {
		declared["name"] = "agent-" + s.utils.uuid()
	}
	if _, hashed := declared["hash"]; !hashed {
		declared["hash"] = s.utils.uuid()
	}
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sync", declared)
	var result SyncAgentResult
	_ = json.Unmarshal(payload, &result)
	var refused ErrorResponse
	_ = json.Unmarshal(payload, &refused)
	return status, result.Config, refused.Error.Message
}

// syncedIn is an agent synced into region running model, which it must be stored as.
func (s *RegionsSuite) syncedIn(region, model string) AgentConfig {
	status, config, refusal := s.synced(map[string]any{"region": region, "llm": model, "mode": "text"})
	s.Require().Equal(http.StatusOK, status, refusal)
	s.Require().Equal(region, config.Region)
	return config
}

// runOn runs the config once and answers the status, what came back and the refusal.
func (s *RegionsSuite) runOn(configID string) (int, Run, string) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/runs", CreateRunRequest{ConfigId: configID, Input: "hello"})
	var run Run
	_ = json.Unmarshal(payload, &run)
	var refused ErrorResponse
	_ = json.Unmarshal(payload, &refused)
	return status, run, refused.Error.Message
}

// named is the app's agent of that name, and false when it has none.
func (s *RegionsSuite) named(name string) (AgentConfig, bool) {
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name="+name, nil, &listed))
	if len(listed) == 0 {
		return AgentConfig{}, false
	}
	return listed[0], true
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

func (s *RegionsSuite) TestAnAgentSyncedInTheEUIsServedOnlyByTheEUInstanceAndBilledAtItsEntry() {
	agent := s.syncedIn(pricing.RegionEU, haiku)
	usAsked := s.us.asked()

	status, run, refusal := s.runOn(agent.Id)

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal("from the eu", run.FinalText)
	s.Equal("eu-key", s.eu.lastKey(), "the eu instance's own key")
	s.Equal(usAsked, s.us.asked(), "the us instance was never asked")
	row := s.rowsOf(anthropic.ProviderName)[0]
	s.Equal("llm-anthropic-claude-haiku-5-5-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.LLM(pricing.RegionEU, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestAnAgentThatNamesNoRegionIsInTheUS() {
	status, agent, refusal := s.synced(map[string]any{"llm": haiku, "mode": "text"})
	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal(pricing.RegionUS, agent.Region)

	status, run, refusal := s.runOn(agent.Id)

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal("from the us", run.FinalText)
	s.Equal("us-key", s.us.lastKey())
	row := s.rowsOf(anthropic.ProviderName)[0]
	s.Equal("llm-anthropic-claude-haiku-5-5-us", row.SKU)
	s.Equal(s.pricedAt(pricingtest.LLM(pricing.RegionUS, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestASyncLeavingTheRegionOutKeepsTheStoredOne() {
	agent := s.syncedIn(pricing.RegionEU, haiku)

	status, again, refusal := s.synced(map[string]any{"name": agent.Name, "instructions": "Be brief."})

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal(agent.Id, again.Id)
	s.Equal(pricing.RegionEU, again.Region)
	s.Equal("Be brief.", value(again.Instructions))
}

func (s *RegionsSuite) TestASyncNamingModelsWithNoInstanceInItsRegionIsRefusedNamingEachBeforeAnythingIsWritten() {
	status, _, refusal := s.synced(map[string]any{"name": "far-" + s.utils.uuid(), "region": pricing.RegionEU,
		"llm": haiku, "subagent": "counted/counted-model", "search": "search-fast"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "subagent: no instance of counted/counted-model in region eu")
	s.Contains(refusal, "search: no instance of search-fast in region eu")
	s.NotContains(refusal, haiku, "haiku has an eu instance")
}

func (s *RegionsSuite) TestARefusedSyncStoresNothing() {
	name := "far-" + s.utils.uuid()

	status, _, _ := s.synced(map[string]any{"name": name, "region": pricing.RegionEU, "llm": "counted/counted-model",
		"knowledge": []map[string]string{{"source": "refunds.md", "text": "Refunds take thirty days."}}})

	s.Equal(http.StatusBadRequest, status)
	_, stored := s.named(name)
	s.False(stored)
	_, kept, err := s.store.KnowledgePlace(context.Background(), s.customerID(), name)
	s.Require().NoError(err)
	s.False(kept, "its knowledge base was never made")
}

func (s *RegionsSuite) TestARegionOfAnotherFormIsRefused() {
	status, _, _ := s.synced(map[string]any{"region": "EU West"})

	s.Equal(http.StatusBadRequest, status)
}

func (s *RegionsSuite) TestAConfigNamingAModelWithNoInstanceInItsRegionIsRefusedOnACreateAPutAndAPatch() {
	status, refusal := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Region: pointerTo(pricing.RegionEU), Llm: pointerTo("counted/counted-model"),
	})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "llm: no instance of counted/counted-model in region eu")

	var stored AgentConfig
	request := AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Llm: pointerTo("counted/counted-model")}
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &stored))
	s.Equal(pricing.RegionUS, stored.Region)
	request.Region = pointerTo(pricing.RegionEU)
	status, refusal = s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+stored.Id, request)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "no instance of counted/counted-model in region eu")
	status, refusal = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+stored.Id,
		AgentConfigPatch{Region: pointerTo(pricing.RegionEU)})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "no instance of counted/counted-model in region eu")
}

func (s *RegionsSuite) TestAPutLeavingTheRegionOutKeepsTheStoredOne() {
	var stored AgentConfig
	request := AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Region: pointerTo(pricing.RegionEU), Llm: pointerTo(haiku)}
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &stored))

	request.Region = nil
	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+stored.Id, request, &updated))

	s.Equal(pricing.RegionEU, updated.Region, "a client that does not know the region does not move the agent")
}

func (s *RegionsSuite) TestAnEUAgentsKnowledgeIsWrittenInTheEUAndBilledAtItsEntry() {
	name := "docs-" + s.utils.uuid()

	status, agent, refusal := s.synced(map[string]any{"name": name, "region": pricing.RegionEU, "llm": haiku,
		"knowledge": []map[string]string{{"source": "refunds.md", "text": "# Refunds\n\nRefunds take thirty days."}}})

	s.Require().Equal(http.StatusOK, status, refusal)
	s.Equal(name, value(agent.KnowledgeNamespace))
	scoped := knowledge.Scoped(s.customerID(), name)
	written, _ := s.knowledge.stored()
	s.NotEqual(scoped, written, "nothing of an eu agent's is written in the us")
	backend, err := turbopuffer.New(turbopuffer.Options{APIKey: "test-key", BaseURL: s.euKnows.URL})
	s.Require().NoError(err)
	names, err := backend.Namespaces(context.Background(), scoped)
	s.Require().NoError(err)
	s.Equal([]string{scoped}, names)
	row := s.rowsOf("turbopuffer")[0]
	s.Equal("knowledge-turbopuffer-write-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.Knowledge(pricing.RegionEU, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestASyncThatWouldMoveTheAgentsKnowledgeBaseIsRefused() {
	name := "docs-" + s.utils.uuid()
	documents := []map[string]string{{"source": "refunds.md", "text": "Refunds take thirty days."}}
	status, agent, refusal := s.synced(map[string]any{"name": name, "llm": haiku, "knowledge": documents})
	s.Require().Equal(http.StatusOK, status, refusal)

	status, _, refusal = s.synced(map[string]any{"name": name, "region": pricing.RegionEU, "knowledge": documents})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, fmt.Sprintf("the knowledge base %q is kept in us, and a knowledge base is never moved", name))
	stored, _ := s.named(name)
	s.Equal(agent.Id, stored.Id)
	s.Equal(pricing.RegionUS, stored.Region, "the agent stayed where its knowledge is")
}

func (s *RegionsSuite) TestAnEUAgentsSandboxIsMadeOnTheEUTargetAndBilledAtItsEntry() {
	made := len(s.daytona.Creations())
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Region: pointerTo(pricing.RegionEU), Llm: pointerTo("scheduling/scheduling-model"),
		Sandbox: pointerTo(Daytona),
	}, &created))
	s.codeHandsBackAReport()

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/runs", CreateRunRequest{ConfigId: created.Id, Input: "how did I walk?"})

	s.Require().Equal(http.StatusOK, status, string(payload))
	creations := s.daytona.Creations()
	s.Require().Len(creations, made+1)
	s.Equal("eu", creations[made]["target"], "made on the eu's Daytona target")
	row := s.rowsOf("daytona")[0]
	s.Equal("sandbox-daytona-running-class-container-eu", row.SKU)
	s.Equal(s.pricedAt(pricingtest.Sandbox(pricing.RegionEU, false), row), row.CostMicros)
}

func (s *RegionsSuite) TestAnAgentNamingASandboxInARegionWithNoneIsRefused() {
	status, _, refusal := s.synced(map[string]any{"region": "ap", "sandbox": "daytona"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(refusal, "sandbox: this deployment runs no sandboxes in region ap")
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
