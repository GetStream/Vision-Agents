package session

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ToolHosts are the workers that run tools for sessions they did not open. The dispatch pool
// is the one there is.
type ToolHosts interface {
	HostedTools(customerID, configID string) ([]dispatch.Tool, time.Duration)
	RunHosted(ctx context.Context, customerID, configID string, call dispatch.ToolCall) (string, error)
}

// HostTools lets sessions on an agent config use the tools a connected worker runs for it.
// Set once, before sessions are served.
func (m *Manager) HostTools(hosts ToolHosts) { m.hosts = hosts }

// hostedTools adds what workers run for the session's config to what the caller declared,
// and returns the runner that sends those calls to them. A name the caller declared stays
// the caller's: whoever opened the session is the one it asked to run it.
func (m *Manager) hostedTools(spec Spec, sessionID string, declared []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
	if m.hosts == nil || spec.ConfigID == "" {
		return declared, next
	}
	offered, _ := m.hosts.HostedTools(spec.CustomerID, spec.ConfigID)
	taken := map[string]bool{}
	for _, tool := range declared {
		taken[tool.Name] = true
	}
	hosted := map[string]bool{}
	for _, tool := range offered {
		if taken[tool.Name] {
			continue
		}
		hosted[tool.Name] = true
		declared = append(declared, harness.Tool{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters})
	}
	if len(hosted) == 0 {
		return declared, next
	}
	m.logger.Info("offering tools a worker hosts", "session", sessionID, "config", spec.ConfigID, "tools", len(hosted))
	return declared, &hostedRunner{hosts: m.hosts, spec: spec, sessionID: sessionID, names: hosted, next: next}
}

// hostedRunner runs the tools a worker hosts through that worker, and hands the rest on.
//
// The worker's own timeout bounds the call rather than the session's, because the session's
// is the caller's estimate of its own tools: a browser that answers a search in a few
// seconds says so, and a source investigation takes a minute.
type hostedRunner struct {
	hosts     ToolHosts
	spec      Spec
	sessionID string
	names     map[string]bool
	next      agent.ToolRunner
}

func (r *hostedRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if !r.names[call.Name] {
		return r.next.Run(ctx, call)
	}
	output, err := r.hosts.RunHosted(ctx, r.spec.CustomerID, r.spec.ConfigID, dispatch.ToolCall{
		ID: call.ID, SessionID: r.sessionID, Name: call.Name, Arguments: call.Arguments,
	})
	if err != nil {
		return nil, err
	}
	return llm.TextParts(output), nil
}
