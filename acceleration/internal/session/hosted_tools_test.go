package session

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type fakeHosts struct {
	ran        []dispatch.ToolCall
	agents     []string
	everyAgent []dispatch.Tool
}

func (f *fakeHosts) HostedTools(customerID, agentID string) ([]dispatch.Tool, time.Duration) {
	if customerID != "acme" {
		return nil, 0
	}
	if agentID == dispatch.EveryAgent {
		return f.everyAgent, time.Minute
	}
	if agentID != "stream-support" {
		return nil, 0
	}
	return []dispatch.Tool{
		{Name: "investigate_sdk", Description: "Read SDK source"},
		{Name: "search_docs", Description: "The worker's search"},
	}, time.Minute
}

func (f *fakeHosts) RunHosted(_ context.Context, _, agentID string, call dispatch.ToolCall) ([]llm.ContentPart, error) {
	f.ran = append(f.ran, call)
	f.agents = append(f.agents, agentID)
	return llm.TextParts("from the worker"), nil
}

type callerTools struct{ ran []string }

func (c *callerTools) Run(_ context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	c.ran = append(c.ran, call.Name)
	return llm.TextParts("from the caller"), nil
}

func TestASessionNamingAHostedAgentIsOfferedTheWorkersTools(t *testing.T) {
	hosts := &fakeHosts{}
	m := &Manager{logger: slog.Default(), hosts: hosts}
	caller := &callerTools{}
	declared := []harness.Tool{{Name: "search_docs", Description: "The browser's search"}}

	// The shape a browser opens: an agent id, no stored config, and its own search.
	spec := Spec{CustomerID: "acme", AgentID: "stream-support"}
	tools, runner := m.hostedTools(spec, newToolLoading(spec), "session-1", nil, declared, caller)

	if len(tools) != 2 || tools[0].Description != "The browser's search" || tools[1].Name != "investigate_sdk" {
		t.Fatalf("offered %+v", tools)
	}
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "investigate_sdk"}); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "2", Name: "search_docs"}); err != nil {
		t.Fatal(err)
	}
	if len(hosts.ran) != 1 || hosts.ran[0].SessionID != "session-1" {
		t.Errorf("the worker ran %+v", hosts.ran)
	}
	if len(caller.ran) != 1 || caller.ran[0] != "search_docs" {
		t.Errorf("a tool the caller declared was not left to the caller: %v", caller.ran)
	}
}

func TestASessionOpenedByAgentNameIsOfferedTheWorkersTools(t *testing.T) {
	hosts := &fakeHosts{}
	m := &Manager{logger: slog.Default(), hosts: hosts}

	// The shape a server SDK opens: a stored config found by name, and an agent id of its own.
	spec := Spec{
		CustomerID: "acme", AgentID: "01JC-generated", ConfigID: "cfg-1", AgentName: "stream-support",
		ToolGroups: []store.ToolGroup{{Name: "support", Tools: []string{"investigate_sdk", "search_docs"}}},
	}
	tools, runner := m.hostedTools(spec, newToolLoading(spec), "session-1", nil, nil, &callerTools{})

	if len(tools) != 2 || tools[0].Name != "investigate_sdk" {
		t.Fatalf("offered %+v", tools)
	}
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "investigate_sdk"}); err != nil {
		t.Fatal(err)
	}
	if len(hosts.agents) != 1 || hosts.agents[0] != "stream-support" {
		t.Errorf("the call went to the workers hosting %v", hosts.agents)
	}
}

func TestASessionNamingAnotherAgentIsOfferedNothingHosted(t *testing.T) {
	m := &Manager{logger: slog.Default(), hosts: &fakeHosts{}}
	caller := &callerTools{}

	sales := Spec{CustomerID: "acme", AgentID: "sales"}
	tools, runner := m.hostedTools(sales, newToolLoading(sales), "s", nil, nil, caller)
	if len(tools) != 0 || runner != caller {
		t.Fatalf("offered %+v", tools)
	}
	byConfig := Spec{CustomerID: "acme", ConfigID: "stream-support"}
	tools, _ = m.hostedTools(byConfig, newToolLoading(byConfig), "s", nil, nil, caller)
	if len(tools) != 0 {
		t.Fatalf("a session naming the agent only as a config was offered %+v", tools)
	}
}

// hostedConversation is a session as the worker running one of its calls reaches it.
type hostedConversation struct{ dispatch.Progress }

func TestAHostedCallCarriesTheConversationItWasMadeIn(t *testing.T) {
	hosts := &fakeHosts{}
	m := &Manager{logger: slog.Default(), hosts: hosts}
	conversation := hostedConversation{}

	spec := Spec{CustomerID: "acme", AgentID: "stream-support"}
	_, runner := m.hostedTools(spec, newToolLoading(spec), "session-1", conversation, nil, nil)
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "investigate_sdk"}); err != nil {
		t.Fatal(err)
	}

	if len(hosts.ran) != 1 || hosts.ran[0].Progress != conversation {
		t.Errorf("the worker cannot reach the conversation: %+v", hosts.ran)
	}
}

func TestAWorkersToolNoGroupNamesIsNotOffered(t *testing.T) {
	m := &Manager{logger: slog.Default(), hosts: &fakeHosts{}}
	spec := Spec{
		CustomerID: "acme", AgentID: "01JC-generated", ConfigID: "cfg-1", AgentName: "stream-support",
		ToolGroups: []store.ToolGroup{{Name: "docs", Tools: []string{"search_docs"}}},
	}

	tools, _ := m.hostedTools(spec, newToolLoading(spec), "session-1", nil, nil, &callerTools{})

	if len(tools) != 1 || tools[0].Name != "search_docs" {
		t.Fatalf("offered %+v", tools)
	}
}

func TestAToolHostedForEveryAgentIsOfferedWhereAToolGroupNamesIt(t *testing.T) {
	hosts := &fakeHosts{everyAgent: []dispatch.Tool{
		{Name: "weather_lookup", Description: "The app's weather"},
		{Name: "refund_order", Description: "The app's refunds"},
	}}
	m := &Manager{logger: slog.Default(), hosts: hosts}

	weather := Spec{
		CustomerID: "acme", AgentID: "01JC-generated", ConfigID: "cfg-1", AgentName: "concierge",
		ToolGroups: []store.ToolGroup{{Name: "outside", Tools: []string{"weather_lookup"}}},
	}
	tools, runner := m.hostedTools(weather, newToolLoading(weather), "session-1", nil, nil, &callerTools{})
	if len(tools) != 1 || tools[0].Name != "weather_lookup" {
		t.Fatalf("offered %+v", tools)
	}
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "weather_lookup"}); err != nil {
		t.Fatal(err)
	}
	if len(hosts.agents) != 1 || hosts.agents[0] != dispatch.EveryAgent {
		t.Errorf("the call went to the workers hosting %q, not those hosting for every agent", hosts.agents)
	}

	unconfigured := Spec{CustomerID: "acme", AgentID: "browser-agent"}
	if tools, _ := m.hostedTools(unconfigured, newToolLoading(unconfigured), "s", nil, nil, &callerTools{}); len(tools) != 0 {
		t.Errorf("a session with no config to enable them was offered %+v", tools)
	}
}

func TestAnAgentsOwnHostedToolWinsOverOneHostedForEveryAgent(t *testing.T) {
	hosts := &fakeHosts{everyAgent: []dispatch.Tool{{Name: "search_docs", Description: "The app's search"}}}
	m := &Manager{logger: slog.Default(), hosts: hosts}
	spec := Spec{
		CustomerID: "acme", AgentID: "01JC-generated", ConfigID: "cfg-1", AgentName: "stream-support",
		ToolGroups: []store.ToolGroup{{Name: "docs", Tools: []string{"search_docs"}}},
	}

	tools, runner := m.hostedTools(spec, newToolLoading(spec), "session-1", nil, nil, &callerTools{})
	if len(tools) != 1 || tools[0].Description != "The worker's search" {
		t.Fatalf("offered %+v", tools)
	}
	if _, err := runner.Run(context.Background(), llm.ToolCall{ID: "1", Name: "search_docs"}); err != nil {
		t.Fatal(err)
	}
	if len(hosts.agents) != 1 || hosts.agents[0] != "stream-support" {
		t.Errorf("the call went to the workers hosting %q", hosts.agents)
	}
}
