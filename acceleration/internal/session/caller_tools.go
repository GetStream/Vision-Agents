package session

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrToolTaken is a tool named as one the session offers that is not its caller's: the
// router's own, a connector's, a worker's or another agent's.
var ErrToolTaken = errors.New("session: the session offers a tool by that name that is not its caller's")

// ErrNoCallerTool is a tool to withdraw that the session's caller does not have.
var ErrNoCallerTool = errors.New("session: the session's caller has no tool by that name")

// AddTools gives the session's caller more tools while the session runs, as though they had
// been declared when it opened, and their calls go to whoever runs the caller's tools. One
// named as a tool the caller has replaces it. It waits for the turn in progress and every call
// in flight to finish (agent.SetTools), and the model is offered them from the next turn.
func (s *Session) AddTools(ctx context.Context, tools []harness.Tool) error {
	return s.changeTools(ctx, func(declared []harness.Tool) ([]harness.Tool, error) {
		next := slices.Clone(declared)
		for _, tool := range tools {
			if at := slices.IndexFunc(next, named(tool.Name)); at >= 0 {
				next[at] = tool
			} else {
				next = append(next, tool)
			}
		}
		return next, nil
	}, func(added map[string]bool) {
		for _, tool := range tools {
			added[tool.Name] = true
		}
	})
}

// RemoveTool withdraws one of the caller's tools. It waits as AddTools does, so once it
// returns no call of the tool is in flight and the model is no longer offered it.
func (s *Session) RemoveTool(ctx context.Context, name string) error {
	return s.changeTools(ctx, func(declared []harness.Tool) ([]harness.Tool, error) {
		at := slices.IndexFunc(declared, named(name))
		if at < 0 {
			return nil, stack.Wrap(fmt.Errorf("%w: %s", ErrNoCallerTool, name))
		}
		return slices.Delete(slices.Clone(declared), at, at+1), nil
	}, func(added map[string]bool) {
		delete(added, name)
	})
}

// changeTools replaces the caller's tools with what change makes of them, and marks which of
// them were added while the session ran. The agent takes them first, so a model that cannot
// change its tools mid-call leaves the session as it was, and the session's own record of
// them changes while the agent holds the floor, so no call is routed by the record before.
func (s *Session) changeTools(ctx context.Context, change func([]harness.Tool) ([]harness.Tool, error), mark func(map[string]bool)) error {
	// Taken as a settings change takes it, so a swap to or from voice builds its agent with
	// the tools this leaves.
	moved, err := s.move(ctx)
	if err != nil {
		return err
	}
	defer moved()
	if s.State() == Ended {
		return stack.Wrap(errors.New("session: the session has ended"))
	}
	next, err := change(s.declaredTools())
	if err != nil {
		return err
	}
	for _, tool := range next {
		if slices.ContainsFunc(s.otherTools, named(tool.Name)) {
			return stack.Wrap(fmt.Errorf("%w: %s", ErrToolTaken, tool.Name))
		}
	}
	if err := s.current().SetTools(ctx, harness.Tools{Tools: s.offering(next)}, func() {
		s.labelMu.Lock()
		s.spec.Tools = next
		if s.addedTools == nil {
			s.addedTools = map[string]bool{}
		}
		mark(s.addedTools)
		s.labelMu.Unlock()
		s.tools.setTimeouts(next)
	}); err != nil {
		return err
	}
	if s.persisted != nil {
		s.persisted.DescribeTools(toolDisplays(next))
	}
	if s.records != nil {
		s.afterRow(func() { s.records.Offered(s.id, toolRows(s)) })
	}
	return nil
}

// declaredTools are the caller's own tools as they are now.
func (s *Session) declaredTools() []harness.Tool {
	s.labelMu.Lock()
	defer s.labelMu.Unlock()
	return s.spec.Tools
}

// callerTool reports whether the caller has a tool called name, and whether it was added
// while the session ran (AddTools).
func (s *Session) callerTool(name string) (declared, added bool) {
	s.labelMu.Lock()
	defer s.labelMu.Unlock()
	return slices.ContainsFunc(s.spec.Tools, named(name)), s.addedTools[name]
}

// offering is everything the session offers with declared as the caller's tools, which go
// first as they did when it opened.
func (s *Session) offering(declared []harness.Tool) []harness.Tool {
	return slices.Concat(declared, s.otherTools)
}

// toolPolicy is next with the caller's tools that asked to be cancelled by an interruption,
// read when a call is made since the caller's tools change while the session runs.
func (s *Session) toolPolicy(next func(string) agent.ToolPolicy) func(string) agent.ToolPolicy {
	return func(name string) agent.ToolPolicy {
		if policy := interruptible(s.declaredTools(), next); policy != nil {
			return policy(name)
		}
		if next != nil {
			return next(name)
		}
		return agent.ToolPolicy{}
	}
}

// toolDisplays is how a persistent conversation shows the calls of the caller's tools.
func toolDisplays(tools []harness.Tool) map[string]persistent.ToolDisplay {
	displays := map[string]persistent.ToolDisplay{}
	for _, tool := range tools {
		display := persistent.ToolDisplay{Title: tool.DisplayTitle, Client: tool.Client}
		if approval := tool.Approval; approval != nil {
			display.Approval = &persistent.ToolApproval{
				Title: approval.Title, Message: approval.Message, ReasonArgument: approval.ReasonArgument,
				AllowTitle: approval.AllowTitle, DeclineTitle: approval.DeclineTitle,
			}
		}
		displays[tool.Name] = display
	}
	return displays
}

func named(name string) func(harness.Tool) bool {
	return func(tool harness.Tool) bool { return tool.Name == name }
}
