package session

import (
	"context"
	"errors"
	"log/slog"
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
// reaches the session through conversation. A tool no group of the session's config names
// is left out: the worker connected before the session opened, so there is nobody to refuse.
//
// Tools hosted for every agent (dispatch.EveryAgent) come after the agent's own, and only
// where a group of the session's config names them: registered once for the app, each agent
// takes the ones its agent.yaml enables, so the others are not worth a warning.
func (m *Manager) hostedTools(spec Spec, loading toolLoading, sessionID string, conversation dispatch.Progress, declared []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
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
	// hosted is the agent id each offered tool is hosted under, which its calls go to.
	hosted := map[string]string{}
	offer := func(agentID string, tool dispatch.Tool) {
		taken[tool.Name] = true
		hosted[tool.Name] = agentID
		offered := harness.Tool{
			Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters,
			DisplayTitle: tool.DisplayTitle, Interruptible: tool.Interruptible, Timeout: tool.Timeout,
		}
		if tool.Approval != "" {
			offered.Approval = &harness.ToolApproval{Title: tool.Approval}
		}
		declared = append(declared, offered)
	}
	var ungrouped []string
	for _, tool := range offered {
		if taken[tool.Name] {
			continue
		}
		if !loading.grouped(tool.Name) {
			ungrouped = append(ungrouped, tool.Name)
			continue
		}
		offer(hostedAs, tool)
	}
	if loading.configured {
		everyAgent, _ := m.hosts.HostedTools(spec.CustomerID, dispatch.EveryAgent)
		for _, tool := range everyAgent {
			if !taken[tool.Name] && loading.grouped(tool.Name) {
				offer(dispatch.EveryAgent, tool)
			}
		}
	}
	if len(ungrouped) > 0 {
		m.logger.Warn("not offering tools a worker hosts that no tool group of the agent's config names",
			"session", sessionID, "agent", hostedAs, "tools", ungrouped)
	}
	if len(hosted) == 0 {
		return declared, next
	}
	m.logger.Info("offering tools a worker hosts", "session", sessionID, "agent", hostedAs, "tools", len(hosted))
	return declared, &hostedRunner{
		hosts: m.hosts, spec: spec, sessionID: sessionID, conversation: conversation, hostedAs: hosted, next: next,
		logger: m.logger,
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
	sessionID    string
	conversation dispatch.Progress
	// hostedAs is the agent id each hosted tool's calls go to.
	hostedAs map[string]string
	next     agent.ToolRunner
	logger   *slog.Logger
}

func (r *hostedRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	agentID, ok := r.hostedAs[call.Name]
	if !ok {
		return r.next.Run(ctx, call)
	}
	hosted := dispatch.ToolCall{
		ID: call.ID, SessionID: r.sessionID, TurnID: call.TurnID, Name: call.Name, Arguments: call.Arguments,
		Approved:     agent.Approved(ctx),
		Conversation: &dispatch.Conversation{CID: r.spec.ConversationID, Custom: r.spec.Custom},
		Progress:     r.conversation,
	}
	if requester, ok := agent.RequesterFrom(ctx); ok {
		hosted.User = &dispatch.User{ID: requester.UserID, Name: requester.Name}
	} else if r.spec.Caller.UserID != "" {
		hosted.User = &dispatch.User{ID: r.spec.Caller.UserID}
	}
	parts, err := r.hosts.RunHosted(ctx, r.spec.CustomerID, agentID, hosted)
	if gone := (dispatch.NoWorkerError{}); errors.As(err, &gone) {
		r.logger.Warn("a tool a worker hosted when the session opened has no worker running it now; the agent carries on without it",
			"session", r.sessionID, "agent", agentID, "tool", call.Name, "tool_call", call.ID)
	}
	return parts, err
}

// hostedProgress is a session as the worker running one of its hosted calls reaches it. The
// call is the agent's, not the session's bridge's, which waits only on calls its watchers run.
type hostedProgress struct{ session *Session }

func (p hostedProgress) HoldTool(id string) bool { return p.session.current().HoldTool(id) }

func (p hostedProgress) ToolUpdate(id, text string, verbatim bool, custom map[string]any) {
	p.session.current().ToolUpdate(id, text, verbatim, custom)
}
