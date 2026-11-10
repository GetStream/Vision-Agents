package api

import (
	"context"
	"fmt"
	"regexp"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/danielgtaylor/huma/v2"
)

// childToolName is what a model provider takes as a tool's name.
var childToolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// ChildAgent is the ChildAgent schema.
type ChildAgent struct {
	Config      string  `json:"config" minLength:"1" doc:"The agent config to hand work to: one of the app's text agents, by name or id. It is stored, and read back, by name."`
	Tool        string  `json:"tool" pattern:"^[a-zA-Z0-9_-]{1,64}$" patternDescription:"up to 64 letters, digits, underscores and hyphens" doc:"The name the model calls it by, unique among the config's agents."`
	Description *string `json:"description,omitempty" doc:"What the model is told the agent does, which is the whole of how it decides when to call it."`
	TimeoutMs   *int    `json:"timeout_ms,omitempty" minimum:"1000" maximum:"600000" doc:"How long one call may take, two minutes when left out. A call never outlasts the run that made it: it gets this or what is left of the run, whichever is less. A call that runs out of time, or whose agent fails, is answered to the model as a failed tool call, and the run goes on."`
}

// childAgentsOf is agents as stored: nil when there are none.
func childAgentsOf(sent *[]ChildAgent) []store.ChildAgent {
	if sent == nil || len(*sent) == 0 {
		return nil
	}
	stored := make([]store.ChildAgent, 0, len(*sent))
	for _, child := range *sent {
		stored = append(stored, store.ChildAgent{
			Config: child.Config, Tool: child.Tool, Description: value(child.Description), TimeoutMs: value(child.TimeoutMs),
		})
	}
	return stored
}

// renderedChildAgents is agents for the wire: absent when there are none.
func renderedChildAgents(stored []store.ChildAgent) *[]ChildAgent {
	if len(stored) == 0 {
		return nil
	}
	rendered := make([]ChildAgent, 0, len(stored))
	for _, child := range stored {
		one := ChildAgent{Config: child.Config, Tool: child.Tool, Description: optional(child.Description)}
		if child.TimeoutMs > 0 {
			timeout := child.TimeoutMs
			one.TimeoutMs = &timeout
		}
		rendered = append(rendered, one)
	}
	return &rendered
}

// childAgentsComplaint refuses agents a config may not hand work to, and names each by the
// config's name as it is stored. Children are one level deep: a parent is a text agent, and a
// child is another of the app's text agents with no agents of its own. That is held from both
// sides, so an agent another one calls can neither gain agents nor stop being a text agent.
// existing is the config before this save, nil for a new one. A list the save leaves as it was
// is not asked about again, as llmFallbacksComplaint leaves a list.
func (s *Server) childAgentsComplaint(ctx context.Context, customerID string, config *store.AgentConfig, existing *store.AgentConfig) (string, bool, error) {
	var was []store.ChildAgent
	wasText := false
	if existing != nil {
		was, wasText = existing.Agents, existing.Mode == store.AgentModeText
	}
	if len(config.Agents) > 0 && config.Mode != store.AgentModeText {
		return "agents are only for text agents: set mode to text", false, nil
	}
	if !slices.Equal(config.Agents, was) {
		tools := map[string]bool{}
		for i, child := range config.Agents {
			if !childToolName.MatchString(child.Tool) {
				return fmt.Sprintf("agents[%d].tool %q is not up to 64 letters, digits, underscores and hyphens", i, child.Tool), false, nil
			}
			if tools[child.Tool] {
				return fmt.Sprintf("agents has two called %q", child.Tool), false, nil
			}
			tools[child.Tool] = true
			found, err := s.configFor(ctx, customerID, nil, &child.Config)
			if err != nil {
				found, err = s.configFor(ctx, customerID, &child.Config, nil)
			}
			switch {
			case err != nil:
				return fmt.Sprintf("agents[%d]: the app has no agent config %q", i, child.Config), false, nil
			case found.ID == config.ID || found.Name == config.Name:
				return fmt.Sprintf("agents[%d]: an agent cannot hand work to itself", i), false, nil
			case found.Mode != store.AgentModeText:
				return fmt.Sprintf("agents[%d]: %q is not a text agent", i, found.Name), false, nil
			case len(found.Agents) > 0:
				return fmt.Sprintf("agents[%d]: %q has agents of its own, and agents are one level deep", i, found.Name), false, nil
			}
			config.Agents[i].Config = found.Name
		}
	}
	renamed := existing != nil && existing.Name != config.Name
	// Only a save that could break a parent reads the app's other configs.
	if len(config.Agents) == 0 && !renamed && !(wasText && config.Mode != store.AgentModeText) {
		return "", true, nil
	}
	name := config.Name
	if renamed {
		name = existing.Name
	}
	parent, err := s.parentOf(ctx, customerID, name, config.ID)
	switch {
	case err != nil:
		return "", false, err
	case parent == "":
		return "", true, nil
	case renamed:
		return fmt.Sprintf("%q hands work to this agent by its name, so it keeps it: take it out of %q's agents first", parent, parent), false, nil
	case len(config.Agents) > 0:
		return fmt.Sprintf("%q hands work to this agent, so it cannot have agents of its own: agents are one level deep", parent), false, nil
	}
	return fmt.Sprintf("%q hands work to this agent, so it stays a text agent", parent), false, nil
}

// parentOf is the name of a config of the app, other than self, that hands work to the
// config called name, and empty when none does.
func (s *Server) parentOf(ctx context.Context, customerID, name, self string) (string, error) {
	others, err := s.store.CustomerAgentConfigs(ctx, customerID)
	if err != nil {
		return "", err
	}
	for _, other := range others {
		if other.ID != self && slices.ContainsFunc(other.Agents, func(child store.ChildAgent) bool { return child.Config == name }) {
			return other.Name, nil
		}
	}
	return "", nil
}

func (*ChildAgent) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Another of the app's text agents a text agent may hand work to, offered to its model as a tool."
	return schema
}
