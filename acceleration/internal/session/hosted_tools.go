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
	HostedTools(customerID, agentID string) ([]dispatch.Tool, time.Duration)
	RunHosted(ctx context.Context, customerID, agentID string, call dispatch.ToolCall) ([]llm.ContentPart, error)
}

// HostTools lets sessions under an agent id use the tools a connected worker runs for it.
// Set once, before sessions are served.
func (m *Manager) HostTools(hosts ToolHosts) { m.hosts = hosts }

// hostedTools adds what workers run for the session's agent to what the caller declared,
// and returns the runner that sends those calls to them. A name the caller declared stays
// the caller's: whoever opened the session is the one it asked to run it.
//
// The agent is the session's agent id, or failing that the name its config was opened by,
// since a session opened by name carries an agent id of its own. The worker running a call
// reaches the session through conversation.
func (m *Manager) hostedTools(spec Spec, sessionID string, conversation dispatch.Progress, declared []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
	if m.hosts == nil {
		return declared, next
	}
	var offered []dispatch.Tool
	hostedAs := ""
	for _, agentID := range []string{spec.AgentID, spec.AgentName} {
		if agentID == "" {
			continue
		}
		if offered, _ = m.hosts.HostedTools(spec.CustomerID, agentID); len(offered) > 0 {
			hostedAs = agentID
			break
		}
	}
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
		declared = append(declared, harness.Tool{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters,
			Interruptible: tool.Interruptible, Timeout: tool.Timeout,
		})
	}
	if len(hosted) == 0 {
		return declared, next
	}
	m.logger.Info("offering tools a worker hosts", "session", sessionID, "agent", hostedAs, "tools", len(hosted))
	return declared, &hostedRunner{
		hosts: m.hosts, spec: spec, agentID: hostedAs, sessionID: sessionID, conversation: conversation, names: hosted, next: next,
	}
}

// hostedRunner runs the tools a worker hosts through that worker, and hands the rest on.
//
// The worker's own timeout bounds the call rather than the session's, because the session's
// is the caller's estimate of its own tools: a browser that answers a search in a few
// seconds says so, and a source investigation takes a minute.
type hostedRunner struct {
	hosts        ToolHosts
	spec         Spec
	agentID      string
	sessionID    string
	conversation dispatch.Progress
	names        map[string]bool
	next         agent.ToolRunner
}

func (r *hostedRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	if !r.names[call.Name] {
		return r.next.Run(ctx, call)
	}
	hosted := dispatch.ToolCall{
		ID: call.ID, SessionID: r.sessionID, TurnID: call.TurnID, Name: call.Name, Arguments: call.Arguments,
		Conversation: &dispatch.Conversation{CID: r.spec.ConversationID, Custom: r.spec.Custom},
		Progress:     r.conversation,
	}
	if requester, ok := agent.RequesterFrom(ctx); ok {
		hosted.User = &dispatch.User{ID: requester.UserID, Name: requester.Name}
	} else if r.spec.Caller.UserID != "" {
		hosted.User = &dispatch.User{ID: r.spec.Caller.UserID}
	}
	return r.hosts.RunHosted(ctx, r.spec.CustomerID, r.agentID, hosted)
}

// hostedProgress is a session as the worker running one of its hosted calls reaches it. The
// call is the agent's, not the session's bridge's, which waits only on calls its watchers run.
type hostedProgress struct{ session *Session }

func (p hostedProgress) HoldTool(id string) bool { return p.session.current().HoldTool(id) }

func (p hostedProgress) ToolUpdate(id, text string, verbatim bool) {
	p.session.current().ToolUpdate(id, text, verbatim)
}
