package session

import (
	"cmp"
	"errors"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ErrToolNotConfigured is a function a session under an agent config was asked to offer that
// none of the config's tool groups names.
var ErrToolNotConfigured = errors.New("session: no tool group of the agent's config names the tool")

// toolLoading is how a session offers each of its tools: a function an SDK provides by the
// config's tool group naming it, a connector's tool by its binding, and the router's own
// tools whole.
type toolLoading struct {
	// configured is whether the session runs under an agent config, whose groups are then the
	// whole of the functions it may offer. One opened without a config has nowhere to name
	// them, and offers what its caller declares.
	configured bool
	functions  map[string]store.ToolGroup
	bindings   map[string]store.ConnectorBinding
}

func newToolLoading(spec Spec) toolLoading {
	loading := toolLoading{
		configured: spec.ConfigID != "",
		functions:  map[string]store.ToolGroup{},
		bindings:   map[string]store.ConnectorBinding{},
	}
	for _, group := range spec.ToolGroups {
		for _, name := range group.Tools {
			loading.functions[strings.TrimSpace(name)] = group
		}
	}
	for _, binding := range spec.ConnectorBindings {
		loading.bindings[binding.Name] = binding
	}
	return loading
}

// grouped reports whether the session may offer the function called name. Video frames are
// the session's own to ask a device for, whoever declares the tool.
func (l toolLoading) grouped(name string) bool {
	if !l.configured || name == agent.VideoFramesTool {
		return true
	}
	_, named := l.functions[name]
	return named
}

// ungrouped are the names among tools the session may not offer.
func (l toolLoading) ungrouped(tools []harness.Tool) []string {
	var names []string
	for _, tool := range tools {
		if !l.grouped(tool.Name) {
			names = append(names, tool.Name)
		}
	}
	return names
}

// missing are the functions the config's groups name that none of tools is, sorted: nothing
// on the session runs them.
func (l toolLoading) missing(tools []harness.Tool) []string {
	var names []string
	for name := range l.functions {
		if !slices.ContainsFunc(tools, named(name)) {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return names
}

// mode is how the tool called name is offered: its group's loading, or its connector
// binding's, or full for everything else.
func (l toolLoading) mode(name string) string {
	if group, named := l.functions[name]; named {
		return cmp.Or(group.Loading, store.ToolLoadingFull)
	}
	if binding, bound := l.binding(name); bound {
		return cmp.Or(binding.Loading, store.ToolLoadingFull)
	}
	return store.ToolLoadingFull
}

// retention is how long what the tool called name was given and gave back is kept, as its
// group or connector binding says, before the policies shorten it. Empty is as long as the
// session.
func (l toolLoading) retention(name string) options.Retention {
	if group, named := l.functions[name]; named {
		return group.OutputRetention
	}
	if binding, bound := l.binding(name); bound {
		return binding.OutputRetention
	}
	return ""
}

// binding is the connector binding whose tool name is, offered as <alias>__<tool>.
func (l toolLoading) binding(name string) (store.ConnectorBinding, bool) {
	alias, _, cut := strings.Cut(name, mcp.Separator)
	if !cut {
		return store.ConnectorBinding{}, false
	}
	binding, bound := l.bindings[alias]
	return binding, bound
}

// summarises reports whether any tool the session may offer is offered by a summary.
func (l toolLoading) summarises() bool {
	for _, group := range l.functions {
		if group.Loading == store.ToolLoadingSummary {
			return true
		}
	}
	for _, binding := range l.bindings {
		if binding.Loading == store.ToolLoadingSummary {
			return true
		}
	}
	return false
}

// arrange is tools as the model is offered them, in the order given: a summarised one by its
// first line and the shape of its arguments, a full one whole, and a deferred one held back
// for load_tool.
func (l toolLoading) arrange(tools []harness.Tool) harness.Tools {
	var arranged harness.Tools
	for _, tool := range tools {
		switch l.mode(tool.Name) {
		case store.ToolLoadingSummary:
			arranged.Tools = append(arranged.Tools, briefTool(tool))
		case store.ToolLoadingDeferred:
			arranged.Deferred = append(arranged.Deferred, tool)
		default:
			arranged.Tools = append(arranged.Tools, tool)
		}
	}
	return arranged
}

// ToolLoading is how the session offers the tool called name: store.ToolLoadingFull,
// ToolLoadingSummary or ToolLoadingDeferred.
func (s *Session) ToolLoading(name string) string { return s.loading.mode(name) }

// describeOffer adds to a call's payload how the tool called name is offered and what its
// definition costs every turn that carries it, whole and as a summary: what choosing its
// loading weighs. A tool the session does not describe, such as a built-in, adds nothing.
func (s *Session) describeOffer(payload map[string]any, name string) {
	all := slices.Concat(s.declaredTools(), s.otherTools)
	at := slices.IndexFunc(all, named(name))
	if at < 0 {
		return
	}
	mode := s.loading.mode(name)
	payload["loading"] = mode
	payload["definition_tokens"] = definitionTokens(all[at])
	if mode == store.ToolLoadingSummary {
		payload["summary_tokens"] = definitionTokens(briefTool(all[at]))
	}
}

func definitionTokens(tool harness.Tool) int64 {
	return llm.ToolTokens(llm.Tool{Name: tool.Name, Description: tool.Description, Parameters: tool.Parameters})
}

// summarised is the tool called name as its server or SDK describes it, when it is one the
// session offers by a summary.
func (s *Session) summarised(name string) (harness.Tool, bool) {
	if s.loading.mode(name) != store.ToolLoadingSummary {
		return harness.Tool{}, false
	}
	all := slices.Concat(s.declaredTools(), s.otherTools)
	at := slices.IndexFunc(all, named(name))
	if at < 0 {
		return harness.Tool{}, false
	}
	return all[at], true
}
