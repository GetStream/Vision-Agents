package session

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type fakeHosts struct{ ran []dispatch.ToolCall }

func (f *fakeHosts) HostedTools(customerID, agentID string) ([]dispatch.Tool, time.Duration) {
	if customerID != "acme" || agentID != "stream-support" {
		return nil, 0
	}
	return []dispatch.Tool{
		{Name: "investigate_sdk", Description: "Read SDK source"},
		{Name: "search_docs", Description: "The worker's search"},
	}, time.Minute
}

func (f *fakeHosts) RunHosted(_ context.Context, _, _ string, call dispatch.ToolCall) (string, error) {
	f.ran = append(f.ran, call)
	return "from the worker", nil
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
	tools, runner := m.hostedTools(Spec{CustomerID: "acme", AgentID: "stream-support"}, "session-1", declared, caller)

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

func TestASessionNamingAnotherAgentIsOfferedNothingHosted(t *testing.T) {
	m := &Manager{logger: slog.Default(), hosts: &fakeHosts{}}
	caller := &callerTools{}

	tools, runner := m.hostedTools(Spec{CustomerID: "acme", AgentID: "sales"}, "s", nil, caller)
	if len(tools) != 0 || runner != caller {
		t.Fatalf("offered %+v", tools)
	}
	tools, _ = m.hostedTools(Spec{CustomerID: "acme", ConfigID: "stream-support"}, "s", nil, caller)
	if len(tools) != 0 {
		t.Fatalf("a session naming the agent only as a config was offered %+v", tools)
	}
}
