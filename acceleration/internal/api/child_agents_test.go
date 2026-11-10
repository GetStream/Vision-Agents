//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// fanned is what fanning asks of its three tools, ask_a, ask_b and ask_c, in its first reply.
var fanned = []string{"ask_a", "ask_b", "ask_c"}

// spent is what each reply of the counted models reports.
var spent = llm.Usage{InputTokens: 1000, InputTokensDetails: llm.InputTokensDetails{CachedTokens: 400}, OutputTokens: 20}

// countedPrice prices spent at 1000 micros, as counted/counted-model is.
var countedPrice = routing.Price{PerMillionInputTokens: 1, PerMillionCachedInputTokens: 0.5, PerMillionOutputTokens: 10}

type ChildAgentsSuite struct {
	RouterSuite
	// gathering holds each child's reply until all of fanned are being written at once.
	gathering *gathering
	// parked is where a parked child's request says it was cancelled.
	parked chan struct{}
	// offeredMu guards offered, the tools each request of offering was offered, by name.
	offeredMu sync.Mutex
	offered   [][]string
}

func TestChildAgentsSuite(t *testing.T) {
	runSuite(t, new(ChildAgentsSuite))
}

func (s *ChildAgentsSuite) SetupSuite() {
	// One, so a parent that runs while its children run proves they take no slot of the app's.
	s.runsPerApp = 1
	s.parked = make(chan struct{}, 16)
	s.llmInstances = []routing.ProviderConfig{
		{Provider: "fanning", Model: "fanning-model", Languages: []string{"la"}, Price: countedPrice},
		{Provider: "gathered", Model: "gathered-model", Languages: []string{"la"}, Price: countedPrice},
		{Provider: "parked", Model: "parked-model", Languages: []string{"la"}},
		{Provider: "offering", Model: "offering-model", Languages: []string{"la"}},
	}
	s.llmFactories = func(registry *llmrouter.Registry) {
		// Calls all three tools in its first reply, then recites what came back.
		registry.Register("fanning", func(routing.Spec) (llmrouter.Provider, error) {
			calls := make([]llm.ToolCall, 0, len(fanned))
			for _, tool := range fanned {
				calls = append(calls, llm.ToolCall{ID: store.NewID(), Name: tool, Arguments: `{"input":"review ` + tool + `"}`})
			}
			return &scriptedLLM{recites: true, calls: calls, usage: spent}, nil
		})
		registry.Register("gathered", func(routing.Spec) (llmrouter.Provider, error) {
			return &gatheredLLM{gathering: s.gathering}, nil
		})
		registry.Register("parked", func(routing.Spec) (llmrouter.Provider, error) {
			return &parkedLLM{cancelled: s.parked}, nil
		})
		// Loads the skill called policy, as checking does, and keeps the tools it was offered.
		registry.Register("offering", func(routing.Spec) (llmrouter.Provider, error) {
			return &offeringLLM{scriptedLLM: scriptedLLM{reply: "Checking.", then: "Judged.", calls: []llm.ToolCall{{
				ID: store.NewID(), Name: harness.LoadSkillTool, Arguments: `{"name":"policy"}`,
			}}}, offered: s.offer}, nil
		})
	}
	s.RouterSuite.SetupSuite()
}

func (s *ChildAgentsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.gathering = &gathering{want: len(fanned), arrived: make(chan struct{}), wait: 5 * time.Second}
	s.offeredMu.Lock()
	s.offered = nil
	s.offeredMu.Unlock()
}

func (s *ChildAgentsSuite) TestAParentRunsItsChildrenAtOnceAndReadsEachAnswer() {
	parent := s.parentOf(map[string]string{"ask_a": s.text("gathered/gathered-model"),
		"ask_b": s.text("gathered/gathered-model"), "ask_c": s.text("gathered/gathered-model")}, nil)

	run := s.run(CreateRunRequest{ConfigId: parent, Input: "review this"})

	for _, tool := range fanned {
		s.Contains(run.FinalText, "together: review "+tool,
			"each child answered while the other two were still being written, and the parent read its answer")
	}
	s.NotContains(run.FinalText, "alone:", "no child waited out the others")
}

func (s *ChildAgentsSuite) TestARunsUsageAddsUpItsChildrensAndBreaksThemDownByTool() {
	parent := s.parentOf(map[string]string{"ask_a": s.text("gathered/gathered-model"),
		"ask_b": s.text("gathered/gathered-model"), "ask_c": s.text("gathered/gathered-model")}, nil)

	run := s.run(CreateRunRequest{ConfigId: parent, Input: "review this"})

	one := CallUsage{InputTokens: 1000, CachedInputTokens: 400, OutputTokens: 20, CostMicros: 1000, Requests: 1}
	s.Equal(map[string]CallUsage{"ask_a": one, "ask_b": one, "ask_c": one}, run.Children)
	// The parent's two replies, one asking for the tools and one after them, and a reply of each child.
	s.Equal(CallUsage{InputTokens: 5000, CachedInputTokens: 2000, OutputTokens: 100, CostMicros: 5000, Requests: 5}, run.Usage)
}

func (s *ChildAgentsSuite) TestAChildThatFailsOrRunsOutOfTimeIsAFailedToolCallAndTheParentAnswers() {
	parent := s.parentOf(map[string]string{"ask_a": s.text("echo/echo-model"),
		"ask_b": s.text("failing/failing-model"), "ask_c": s.text("stuck/stuck-model")},
		map[string]int{"ask_c": 1000})

	started := time.Now()
	run := s.run(CreateRunRequest{ConfigId: parent, Input: "review this"})

	s.Less(time.Since(started), 30*time.Second, "the stuck child was given up on at its own timeout")
	s.Contains(run.FinalText, "be brief", "the child that answered is read")
	s.Contains(run.FinalText, "tool: That did not work: failing: the provider is down", "the failed child is a failed tool call")
	s.Contains(run.FinalText, "tool: That did not work: ask_c did not finish in the time it had", "so is the one that ran out of time")
}

func (s *ChildAgentsSuite) TestTheParentsDeadlineCutsItsChildrenShort() {
	parked := s.text("parked/parked-model")
	parent := s.parentOf(map[string]string{"ask_a": parked, "ask_b": parked, "ask_c": parked}, nil)

	started := time.Now()
	status, failure := s.runFails(CreateRunRequest{ConfigId: parent, Input: "review this", TimeoutMs: pointerTo(1500)})

	s.Equal(http.StatusGatewayTimeout, status)
	s.Equal("timeout", failure.Code)
	s.Less(time.Since(started), 10*time.Second, "the run did not wait out its children's two minutes")
	for range fanned {
		select {
		case <-s.parked:
		case <-time.After(settleFor):
			s.Fail("a child was still running after its parent's run ended")
			return
		}
	}
}

func (s *ChildAgentsSuite) TestAnAgentWithNoAgentsIsOfferedTheToolsItAlwaysWas() {
	without := s.offeringConfig(nil)
	s.run(CreateRunRequest{ConfigId: without, Input: "hello"})
	plain := s.takeOffered()

	with := s.offeringConfig(&[]ChildAgent{{Config: s.textNamed("echo/echo-model"), Tool: "ask_a"}})
	s.run(CreateRunRequest{ConfigId: with, Input: "hello"})
	parent := s.takeOffered()

	s.Require().Len(plain, 2, "the reply that loaded the skill and the one after it")
	s.Require().Len(parent, 2)
	s.NotContains(plain[0], "ask_a")
	for i := range plain {
		s.Equal(plain[i], slices.DeleteFunc(slices.Clone(parent[i]), func(name string) bool { return name == "ask_a" }),
			"a child's tool is added and nothing else moves")
	}
	s.Contains(parent[0], "ask_a")
}

func (s *ChildAgentsSuite) TestAnAgentNamesItsChildrenByNameWhateverItWasSent() {
	childID := s.text("echo/echo-model")
	var child AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+childID, nil, &child))

	created := s.saveParent(http.StatusCreated, []ChildAgent{{Config: childID, Tool: "ask", TimeoutMs: pointerTo(5000)}})

	s.Equal([]ChildAgent{{Config: child.Name, Tool: "ask", TimeoutMs: pointerTo(5000)}}, value(created.Agents))
}

func (s *ChildAgentsSuite) TestAnUpdateThatLeavesOutTheAgentsKeepsThemAndAnEmptyListRemovesThem() {
	created := s.saveParent(http.StatusCreated, []ChildAgent{{Config: s.textNamed("echo/echo-model"), Tool: "ask"}})

	var kept AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": created.Name, "mode": "text", "llm": "echo/echo-model"}, &kept))
	s.Len(value(kept.Agents), 1)

	var removed AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": created.Name, "mode": "text", "llm": "echo/echo-model", "agents": []any{}}, &removed))
	s.Nil(removed.Agents)
}

func (s *ChildAgentsSuite) TestAnotherAppsAgentIsNoChild() {
	theirs := s.text("echo/echo-model")
	s.useApp(s.data.createApp())

	s.refused([]ChildAgent{{Config: theirs, Tool: "ask"}}, "the app has no agent config")
}

func (s *ChildAgentsSuite) TestAVoiceAgentIsNoChild() {
	var voice AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "voice-" + s.utils.uuid(), "llm": "echo/echo-model"}, &voice))

	s.refused([]ChildAgent{{Config: voice.Name, Tool: "ask"}}, "is not a text agent")
}

func (s *ChildAgentsSuite) TestAVoiceAgentHasNoChildren() {
	status, failure := s.saveConfig(map[string]any{"name": "voice-" + s.utils.uuid(), "llm": "echo/echo-model",
		"agents": []ChildAgent{{Config: s.textNamed("echo/echo-model"), Tool: "ask"}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "only for text agents")
}

func (s *ChildAgentsSuite) TestAChildWithChildrenOfItsOwnIsRefused() {
	middle := s.saveParent(http.StatusCreated, []ChildAgent{{Config: s.textNamed("echo/echo-model"), Tool: "ask"}})

	s.refused([]ChildAgent{{Config: middle.Name, Tool: "ask"}}, "agents are one level deep")
}

func (s *ChildAgentsSuite) TestAChildCannotHandWorkBackToItsParent() {
	child := s.textNamed("echo/echo-model")
	parent := s.saveParent(http.StatusCreated, []ChildAgent{{Config: child, Tool: "ask"}})

	var found []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name="+child, nil, &found))
	s.Require().Len(found, 1)
	status, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+found[0].Id,
		map[string]any{"agents": []ChildAgent{{Config: parent.Name, Tool: "back"}}})

	s.Equal(http.StatusBadRequest, status, string(payload))
	s.Contains(string(payload), "one level deep")
}

func (s *ChildAgentsSuite) TestAnAgentCannotHandWorkToItself() {
	created := s.saveParent(http.StatusCreated, []ChildAgent{{Config: s.textNamed("echo/echo-model"), Tool: "ask"}})

	status, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"agents": []ChildAgent{{Config: created.Name, Tool: "me"}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), "cannot hand work to itself")
}

func (s *ChildAgentsSuite) TestAChildStaysATextAgent() {
	child := s.textNamed("echo/echo-model")
	s.saveParent(http.StatusCreated, []ChildAgent{{Config: child, Tool: "ask"}})
	var found []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name="+child, nil, &found))
	s.Require().Len(found, 1)

	status, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+found[0].Id, map[string]any{"mode": "voice"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(payload), "stays a text agent")
}

func (s *ChildAgentsSuite) TestAChildKeepsItsNameWhileAParentCallsIt() {
	child := s.textConfig("echo/echo-model")
	s.saveParent(http.StatusCreated, []ChildAgent{{Config: child.Name, Tool: "ask"}})

	status, payload := s.serverClient.call(http.MethodPatch, "/v1/agents/configs/"+child.Id, map[string]any{"name": "renamed-" + s.utils.uuid()})

	s.Equal(http.StatusBadRequest, status, "a parent names its child by name, and another config could take the old one")
	s.Contains(string(payload), "keeps it")
}

func (s *ChildAgentsSuite) TestAChildIsNotDeletedWhileAParentCallsIt() {
	child := s.textConfig("echo/echo-model")
	s.saveParent(http.StatusCreated, []ChildAgent{{Config: child.Name, Tool: "ask"}})

	status, payload := s.serverClient.call(http.MethodDelete, "/v1/agents/configs/"+child.Id, nil)

	s.Equal(http.StatusConflict, status, string(payload))
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+child.Id, nil, nil))
}

func (s *ChildAgentsSuite) TestTwoChildrenCannotShareATool() {
	child := s.textNamed("echo/echo-model")

	s.refused([]ChildAgent{{Config: child, Tool: "ask"}, {Config: child, Tool: "ask"}}, `two called "ask"`)
}

// text is the id of a text agent config of the app answered by model.
func (s *ChildAgentsSuite) text(model string) string {
	return s.textConfig(model).Id
}

// textNamed is the name of a text agent config of the app answered by model.
func (s *ChildAgentsSuite) textNamed(model string) string {
	return s.textConfig(model).Name
}

func (s *ChildAgentsSuite) textConfig(model string) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "agent-" + s.utils.uuid(), "mode": "text", "llm": model, "instructions": "be brief",
	}, &created))
	return created
}

// parentOf is the id of a text agent on fanning whose agents are the configs by tool, each
// with its timeout in timeouts when there is one.
func (s *ChildAgentsSuite) parentOf(children map[string]string, timeouts map[string]int) string {
	var agents []ChildAgent
	for _, tool := range fanned {
		child := ChildAgent{Config: children[tool], Tool: tool, Description: pointerTo("reviews one lens")}
		if timeout, ok := timeouts[tool]; ok {
			child.TimeoutMs = &timeout
		}
		agents = append(agents, child)
	}
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "lead-" + s.utils.uuid(), "mode": "text", "llm": "fanning/fanning-model", "agents": agents,
	}, &created))
	return created.Id
}

// offeringConfig is the id of a text agent on offering, with the policy skill and agents.
func (s *ChildAgentsSuite) offeringConfig(agents *[]ChildAgent) string {
	request := map[string]any{"name": "agent-" + s.utils.uuid(), "mode": "text", "llm": "offering/offering-model",
		"skills": []string{"policy"}}
	if agents != nil {
		request["agents"] = *agents
	}
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &config))
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/skills", map[string]any{
		"config_id": config.Id, "name": "policy", "description": "the labels to judge by", "instructions": "Flag spam.",
	}, nil))
	return config.Id
}

func (s *ChildAgentsSuite) saveParent(status int, agents []ChildAgent) AgentConfig {
	var created AgentConfig
	s.Require().Equal(status, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "lead-" + s.utils.uuid(), "mode": "text", "llm": "echo/echo-model", "agents": agents,
	}, &created))
	return created
}

func (s *ChildAgentsSuite) saveConfig(request map[string]any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/configs", request)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

// refused saves a text parent with agents and requires a 400 saying says.
func (s *ChildAgentsSuite) refused(agents []ChildAgent, says string) {
	status, failure := s.saveConfig(map[string]any{"name": "lead-" + s.utils.uuid(), "mode": "text",
		"llm": "echo/echo-model", "agents": agents})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, says)
}

func (s *ChildAgentsSuite) offer(tools []llm.Tool) {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}
	s.offeredMu.Lock()
	defer s.offeredMu.Unlock()
	s.offered = append(s.offered, names)
}

func (s *ChildAgentsSuite) takeOffered() [][]string {
	s.offeredMu.Lock()
	defer s.offeredMu.Unlock()
	taken := s.offered
	s.offered = nil
	return taken
}

func (s *ChildAgentsSuite) run(body CreateRunRequest) Run {
	var run Run
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/runs", body, &run))
	return run
}

func (s *ChildAgentsSuite) runFails(body CreateRunRequest) (int, ErrorDetail) {
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/runs", body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

// gathering lets each of want requests through only once all of them have arrived, or after
// wait, so a request answers together only when the others were in flight beside it.
type gathering struct {
	mu      sync.Mutex
	want    int
	count   int
	arrived chan struct{}
	wait    time.Duration
}

func (g *gathering) gather(ctx context.Context) string {
	g.mu.Lock()
	g.count++
	if g.count == g.want {
		close(g.arrived)
	}
	g.mu.Unlock()
	select {
	case <-g.arrived:
		return "together"
	case <-time.After(g.wait):
		return "alone"
	case <-ctx.Done():
		return "alone"
	}
}

// gatheredLLM answers once its gathering lets it, with whether it was together and the last
// thing it was asked.
type gatheredLLM struct {
	scriptedLLM
	gathering *gathering
}

func (g *gatheredLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	verdict := g.gathering.gather(ctx)
	asked := ""
	if said := harness.WithoutNotes(params.Input); len(said) > 0 {
		asked = said[len(said)-1].Content
	}
	answer := &scriptedLLM{reply: verdict + ": " + strings.TrimSpace(asked), usage: spent}
	return answer.Create(ctx, params)
}

// parkedLLM never answers, and says so on cancelled once its request is given up.
type parkedLLM struct {
	scriptedLLM
	cancelled chan<- struct{}
}

func (p *parkedLLM) Create(ctx context.Context, _ llm.ResponseParams) (*llm.Stream, error) {
	<-ctx.Done()
	p.cancelled <- struct{}{}
	return nil, ctx.Err()
}

// offeringLLM is a scriptedLLM that hands offered the tools of each request.
type offeringLLM struct {
	scriptedLLM
	offered func([]llm.Tool)
}

func (o *offeringLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	o.offered(params.Tools)
	return o.scriptedLLM.Create(ctx, params)
}
