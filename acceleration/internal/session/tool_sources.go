package session

import "github.com/GetStream/Vision-Agents/acceleration/internal/harness"

// ToolSource is where a tool a session's model is offered came from.
//
// It is what tells a tool that is missing because the worker hosting it is not connected
// from one that was never there: a session's tools are fixed when it opens, so a worker that
// was down then leaves no trace but the absence of its tools.
type ToolSource string

const (
	// ToolFromSession is one whoever opened the session declared, and runs.
	ToolFromSession ToolSource = "session"
	// ToolFromWorker is one a connected worker hosts for the session's agent.
	ToolFromWorker ToolSource = "worker"
	// ToolFromConnector is one a connector binding opened.
	ToolFromConnector ToolSource = "connector"
	// ToolBuiltIn is the router's own: the call tools, search, scheduling, moderation, and
	// whatever the agent adds for its skills and knowledge.
	ToolBuiltIn ToolSource = "builtin"
)

// markTools records where tools came from, leaving a name already recorded as it was.
func markTools(sources map[string]ToolSource, tools []harness.Tool, source ToolSource) {
	for _, tool := range tools {
		if _, taken := sources[tool.Name]; !taken {
			sources[tool.Name] = source
		}
	}
}

// ToolSource says where one of the tools the session's model is offered came from. A name
// the session was not opened with is one the agent added itself, which is built in.
func (s *Session) ToolSource(name string) ToolSource {
	if source, ok := s.toolSources[name]; ok {
		return source
	}
	return ToolBuiltIn
}
