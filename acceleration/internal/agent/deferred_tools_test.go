package agent

import (
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// defersRefunds gives the agent lookup_order to offer and refund_last_order held back.
func (s *AgentSuite) defersRefunds() {
	s.ownsTools("refunded")
	s.tools.Deferred = []harness.Tool{{
		Name:        "refund_last_order",
		Description: "Give the customer their money back for the last order.",
		Parameters:  map[string]any{"type": "object"},
	}}
}

func offeredNames(tools []harness.Tool) []string {
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Name
	}
	return names
}

func toolsOf(requested []llm.Tool) []harness.Tool {
	tools := make([]harness.Tool, len(requested))
	for i, tool := range requested {
		tools[i] = harness.Tool{Name: tool.Name}
	}
	return tools
}

func (s *AgentSuite) TestADeferredToolIsNotOfferedButFindingAndLoadingOneIs() {
	s.defersRefunds()
	s.joinText()

	offered := offeredNames(s.agent.availableTools().Tools)

	s.NotContains(offered, "refund_last_order")
	s.Contains(offered, harness.SearchToolName)
	s.Contains(offered, harness.LoadToolName)
}

func (s *AgentSuite) TestNothingDeferredOffersNoSearchTool() {
	s.ownsTools("")
	s.joinText()

	s.NotContains(offeredNames(s.agent.availableTools().Tools), harness.SearchToolName)
}

func (s *AgentSuite) TestADeferredToolTheModelHasNotLoadedDoesNotRun() {
	s.defersRefunds()
	s.joinText()

	s.model.reply = []string{"Refunding."}
	s.model.then = []string{"Done."}
	s.asksFor("refund_last_order", `{}`)

	_, err := s.agent.RespondTo(s.ctx, "I want my money back", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) >= 1 }, "the model was never asked")
	s.never(func() bool { return len(s.runner.asked()) > 0 }, "a tool the model never loaded ran")
}

func (s *AgentSuite) TestALoadedToolIsOfferedAfterEveryOther() {
	s.defersRefunds()
	s.joinText()
	before := offeredNames(s.agent.availableTools().Tools)

	s.agent.runTool(harness.ToolRequested{TurnID: "turn-1", Call: llm.ToolCall{
		ID: "call-1", Name: harness.LoadToolName, Arguments: `{"names":["refund_last_order"]}`,
	}})

	s.Equal(append(before, "refund_last_order"), offeredNames(s.agent.availableTools().Tools),
		"a load moved the tools offered before it, which costs the provider's cache of them")
}

func (s *AgentSuite) TestALoadedToolRunsWhenTheModelCallsIt() {
	s.defersRefunds()
	s.joinText()
	s.agent.runTool(harness.ToolRequested{TurnID: "turn-1", Call: llm.ToolCall{
		ID: "call-1", Name: harness.LoadToolName, Arguments: `{"names":["refund_last_order"]}`,
	}})
	// The load's result is answered by a reply of its own; the call is the one after it.
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the load was never answered")
	s.model.mu.Lock()
	s.model.calls = []llm.ToolCall{{ID: "call-2", Name: "refund_last_order", Arguments: `{}`}}
	s.model.callsOn = 2
	s.model.then = []string{"Done."}
	s.model.mu.Unlock()

	_, err := s.agent.RespondTo(s.ctx, "I want my money back", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.runner.asked()) == 1 }, "the loaded tool never ran")
	s.Equal("refund_last_order", s.runner.asked()[0].Name)
}

func (s *AgentSuite) TestTheModelIsOfferedWhatItLoadedOnTheNextStepOfTheSameTurn() {
	s.defersRefunds()
	s.joinText()
	s.model.reply = []string{"Let me find that."}
	s.model.then = []string{"Done."}
	s.asksFor(harness.LoadToolName, `{"names":["refund_last_order"]}`)

	_, err := s.agent.RespondTo(s.ctx, "I want my money back", nil, options.LLM{})
	s.Require().NoError(err)

	s.eventually(func() bool { return len(s.model.requests()) >= 2 }, "the model was never asked after it loaded the tool")
	first, second := s.model.requests()[0], s.model.requests()[1]
	s.NotContains(offeredNames(toolsOf(first.Tools)), "refund_last_order")
	s.Equal("refund_last_order", second.Tools[len(second.Tools)-1].Name)
}

func (s *AgentSuite) TestASearchAnswersWithTheDeferredToolsItMatches() {
	s.defersRefunds()
	s.joinText()

	s.agent.runTool(harness.ToolRequested{TurnID: "turn-1", Call: llm.ToolCall{
		ID: "call-1", Name: harness.SearchToolName, Arguments: `{"query":"money back"}`,
	}})

	s.Contains(s.historyWith(llm.ToolResult), `"name":"refund_last_order"`)
}

func (s *AgentSuite) TestAConversationRestoredIsOfferedTheToolsItHadLoaded() {
	s.defersRefunds()
	s.joinText()

	s.agent.RestoreHistory([]llm.Message{
		{Role: llm.User, Content: "I want my money back"},
		{Role: llm.Assistant, ToolCalls: []llm.ToolCall{{
			ID: "call-1", Name: harness.LoadToolName, Arguments: `{"names":["refund_last_order"]}`,
		}}},
		{Role: llm.ToolResult, ToolCallID: "call-1", Content: `{"loaded":[{"name":"refund_last_order"}]}`},
	})

	offered := offeredNames(s.agent.availableTools().Tools)
	s.Equal("refund_last_order", offered[len(offered)-1])
	s.False(strings.Contains(strings.Join(offered, ","), "refund_last_order,refund_last_order"))
}
