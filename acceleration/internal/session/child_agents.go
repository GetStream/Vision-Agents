package session

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxChildrenAtOnce is how many child agents one session runs at once. A call past it waits
// for one to finish, within its own deadline.
const maxChildrenAtOnce = 8

// defaultChildTimeout is how long a call to a child agent may take when its entry names no
// timeout: as long as a run that names none.
const defaultChildTimeout = 2 * time.Minute

// childInput is the one argument a child agent's tool takes.
const childInput = "input"

// childAgents offers a text session the app's other text agents its config names, each as a
// tool. A call runs the child on its input as an incognito run of its own and answers with
// what the child said last. A name any other tool has stays that tool's.
//
// A child's run is held to the parent's: it ends when the parent's tool call does, and never
// after the parent's RunDeadline. It takes none of the app's run slots: the parent's slot
// covers it, and maxChildrenAtOnce bounds how many one parent holds.
func (m *Manager) childAgents(spec Spec, parent *Session, declared []harness.Tool, next agent.ToolRunner) ([]harness.Tool, agent.ToolRunner) {
	if !spec.Text || len(spec.Agents) == 0 || m.options.Configs == nil {
		return declared, next
	}
	// The agent offers these itself, past the tools it is handed.
	taken := map[string]bool{harness.LoadSkillTool: true, sandbox.ToolName: true, "confirm_tool_call": true}
	for _, tool := range declared {
		taken[tool.Name] = true
	}
	runner := &childRunner{
		manager: m, parent: parent, spec: spec, next: next,
		children: map[string]store.ChildAgent{}, slots: make(chan struct{}, maxChildrenAtOnce),
	}
	for _, child := range spec.Agents {
		if taken[child.Tool] {
			m.logger.Warn("a child agent's tool has the name of another tool, which keeps it",
				"config", spec.ConfigID, "tool", child.Tool)
			continue
		}
		taken[child.Tool] = true
		runner.children[child.Tool] = child
		declared = append(declared, harness.Tool{
			Name: child.Tool,
			Description: cmp.Or(strings.TrimSpace(child.Description),
				"Hands work to the "+child.Config+" agent and answers with what it said."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{childInput: map[string]any{
					"type":        "string",
					"description": "Everything the agent needs for the work. It sees nothing else of this conversation.",
				}},
				"required": []string{childInput},
			},
		})
	}
	if len(runner.children) == 0 {
		return declared, next
	}
	return declared, runner
}

// childRunner runs the child agents childAgents offered, and hands every other tool on.
type childRunner struct {
	manager  *Manager
	parent   *Session
	spec     Spec
	next     agent.ToolRunner
	children map[string]store.ChildAgent
	slots    chan struct{}
}

func (r *childRunner) Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error) {
	child, ok := r.children[call.Name]
	if !ok {
		return r.next.Run(ctx, call)
	}
	var arguments map[string]any
	_ = json.Unmarshal([]byte(call.Arguments), &arguments)
	input, _ := arguments[childInput].(string)
	if strings.TrimSpace(input) == "" {
		return nil, fmt.Errorf("%s was called with no %s", call.Name, childInput)
	}

	timeout := defaultChildTimeout
	if child.TimeoutMs > 0 {
		timeout = time.Duration(child.TimeoutMs) * time.Millisecond
	}
	deadline := time.Now().Add(timeout)
	if !r.spec.RunDeadline.IsZero() && r.spec.RunDeadline.Before(deadline) {
		deadline = r.spec.RunDeadline
	}
	ctx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	select {
	case r.slots <- struct{}{}:
		defer func() { <-r.slots }()
	case <-ctx.Done():
		return nil, childEnded(call.Name, ctx.Err())
	}

	config, found, err := r.manager.options.Configs.AgentConfigByName(ctx, r.spec.CustomerID, child.Config)
	if err != nil {
		return nil, err
	}
	// Checked again as the child runs, since it may have changed since the parent was saved.
	switch {
	case !found:
		return nil, fmt.Errorf("the agent %q behind %s no longer exists", child.Config, call.Name)
	case config.Mode != store.AgentModeText:
		return nil, fmt.Errorf("the agent %q behind %s is no text agent", child.Config, call.Name)
	case len(config.Agents) > 0:
		return nil, fmt.Errorf("the agent %q behind %s calls agents of its own", child.Config, call.Name)
	}

	spec := FromConfig(config)
	spec.CustomerID = r.spec.CustomerID
	spec.Caller, spec.CallerKind = r.spec.Caller, r.spec.CallerKind
	// The parent's cost labels go on the child's requests too, so its spend is counted
	// where the parent's is.
	if len(r.spec.Tags) > 0 {
		tags := routing.Tags{}
		maps.Copy(tags, spec.Tags)
		maps.Copy(tags, r.spec.Tags)
		spec.Tags = tags
	}

	run, err := r.manager.Run(ctx, spec, input, nil)
	// Spent is spent, whether or not the child answered.
	r.parent.broadcast(ChildRan{Tool: call.Name, Usage: run.Usage})
	switch {
	case err != nil:
		return nil, err
	case ctx.Err() != nil:
		return nil, childEnded(call.Name, ctx.Err())
	case run.Failed != nil:
		return nil, run.Failed
	case run.FinalText == "":
		return nil, fmt.Errorf("%s gave no answer", call.Name)
	}
	return llm.TextParts(run.FinalText), nil
}

// childEnded is a child given up on before it answered.
func childEnded(tool string, err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%s did not finish in the time it had", tool)
	}
	return fmt.Errorf("%s was stopped before it finished", tool)
}
