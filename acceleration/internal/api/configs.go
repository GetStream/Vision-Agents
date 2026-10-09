package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/danielgtaylor/huma/v2"
)

// errNoConfigs is what the config and skill paths say on a deployment without a database.
// They are stored rather than computed, so there is nothing to serve without one.
var errNoConfigs = notConfigured("agent configs are not available: no database configured")

// mcpBrandingTimeout is how long saving a config waits for its MCP servers to describe
// themselves.
const mcpBrandingTimeout = 5 * time.Second

// listAgentConfigs returns the calling customer's configs, newest first.
func (s *Server) listAgentConfigs(ctx context.Context, request *listAgentConfigsRequest) (*listAgentConfigsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	// A name is resolved through the index rather than by reading every config and filtering
	// here, and an empty answer is an empty list rather than a 404: this is a list endpoint,
	// and a caller looking a name up is asking whether it is there.
	if named := value(request.Name.ptr()); named != "" {
		found, exists, err := s.configs.AgentConfigByName(ctx, customerID, named)
		if err != nil {
			return nil, err
		}
		if !exists {
			return &listAgentConfigsResponse{Body: []AgentConfig{}}, nil
		}
		return &listAgentConfigsResponse{Body: []AgentConfig{agentConfigOf(found)}}, nil
	}

	stored, err := s.store.CustomerAgentConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]AgentConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, agentConfigOf(config))
	}
	return &listAgentConfigsResponse{Body: listed}, nil
}

// createAgentConfig stores a configuration sessions can be created from.
func (s *Server) createAgentConfig(ctx context.Context, request *createAgentConfigRequest) (*createAgentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if message, ok := configComplaint(*request.Body); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok, err := s.unboundConnectors(ctx, customerID, request.Body.Connectors); err != nil {
		return nil, err
	} else if !ok {
		return nil, invalidRequest(message)
	}

	config := storedConfig(*request.Body, customerID)
	if message, ok := textThinkingComplaint(&config, request.Body.ThinkingLlm); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginEventsComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginEntriesComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := mcpServersComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	servers, message, ok := s.describedMCPServers(ctx, config.MCPServers, nil)
	if !ok {
		return nil, invalidRequest(message)
	}
	config.MCPServers = servers
	if message, ok := s.channelsComplaint(ctx, config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginAliasComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if err := s.configs.CreateAgentConfig(ctx, &config); err != nil {
		return nil, storeFailure(err, errAgentNameTaken)
	}
	stored := agentConfigOf(config)
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditAgentConfig, ResourceID: config.ID, ResourceName: config.Name,
		Action: store.AuditCreated, Changes: auditDiff(nil, stored),
	})
	s.pluginEvents.Changed(customerID, config.ID)
	return &createAgentConfigResponse{Body: stored}, nil
}

// getAgentConfig returns one config.
func (s *Server) getAgentConfig(ctx context.Context, request *getAgentConfigRequest) (*getAgentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	return &getAgentConfigResponse{Body: agentConfigOf(config)}, nil
}

// updateAgentConfig replaces a config with what it now is.
func (s *Server) updateAgentConfig(ctx context.Context, request *updateAgentConfigRequest) (*updateAgentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if message, ok := configComplaint(*request.Body); !ok {
		return nil, invalidRequest(message)
	}

	existing, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	if message, ok, err := s.unboundConnectors(ctx, customerID, request.Body.Connectors); err != nil {
		return nil, err
	} else if !ok {
		return nil, invalidRequest(message)
	}

	config := storedConfig(*request.Body, customerID)
	if message, ok := textThinkingComplaint(&config, request.Body.ThinkingLlm); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginEventsComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginEntriesComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := mcpServersComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	servers, message, ok := s.describedMCPServers(ctx, config.MCPServers, existing.MCPServers)
	if !ok {
		return nil, invalidRequest(message)
	}
	config.MCPServers = servers
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	// Unlike the rest of an update, bindings left out are kept rather than cleared: a client
	// written before they existed saves a config without them, and saving it would otherwise
	// take away every tool the agent was granted.
	if request.Body.Connectors == nil {
		config.Connectors = existing.Connectors
	}
	// Kept for the same reason: a client that does not know the setting must not turn the
	// cards off by saving.
	if request.Body.EpisodeCards == nil {
		config.EpisodeCards = existing.EpisodeCards
	}
	if request.Body.ProgressiveTools == nil {
		config.ProgressiveTools = existing.ProgressiveTools
	}
	if message, ok := s.channelsComplaint(ctx, config); !ok {
		return nil, invalidRequest(message)
	}
	if message, ok := pluginAliasComplaint(config); !ok {
		return nil, invalidRequest(message)
	}
	if err := s.configs.UpdateAgentConfig(ctx, &config); err != nil {
		return nil, storeFailure(err, errAgentNameTaken)
	}
	stored := agentConfigOf(config)
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditAgentConfig, ResourceID: config.ID, ResourceName: config.Name,
		Action: store.AuditUpdated, Changes: auditDiff(agentConfigOf(existing), stored),
	})
	s.pluginEvents.Changed(customerID, config.ID)
	return &updateAgentConfigResponse{Body: stored}, nil
}

// deleteAgentConfig stops a config being usable.
func (s *Server) deleteAgentConfig(ctx context.Context, request *deleteAgentConfigRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	// Read before it goes, so the entry recording the deletion can say what was deleted.
	existing, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}
	if err := s.configs.DeleteAgentConfig(ctx, customerID, request.Id); err != nil {
		return nil, errUnknownConfig
	}
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditAgentConfig, ResourceID: existing.ID, ResourceName: existing.Name,
		Action: store.AuditDeleted, Changes: auditDiff(agentConfigOf(existing), nil),
	})
	return nil, nil
}

// listSkills returns the calling customer's skills, newest first, or only the ones
// belonging to one agent config.
func (s *Server) listSkills(ctx context.Context, request *listSkillsRequest) (*listSkillsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	stored, err := s.store.CustomerSkills(ctx, customerID, value(request.ConfigId.ptr()))
	if err != nil {
		return nil, err
	}

	listed := make([]Skill, 0, len(stored))
	for _, skill := range stored {
		listed = append(listed, skillOf(skill))
	}
	return &listSkillsResponse{Body: listed}, nil
}

// createSkill defines a kind of work worth handing to the slower model.
func (s *Server) createSkill(ctx context.Context, request *createSkillRequest) (*createSkillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if message, ok := skillComplaint(*request.Body); !ok {
		return nil, invalidRequest(message)
	}
	if _, err := s.configs.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return nil, errUnknownConfig
	}

	skill := storedSkill(*request.Body, customerID)
	if err := s.configs.CreateSkill(ctx, &skill); err != nil {
		return nil, storeFailure(err, errSkillNameTaken)
	}
	stored := skillOf(skill)
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditSkill, ResourceID: skill.ID, ResourceName: skill.Name,
		AgentID: skill.ConfigID, Action: store.AuditCreated, Changes: auditDiff(nil, stored),
	})
	return &createSkillResponse{Body: stored}, nil
}

// getSkill returns one skill.
func (s *Server) getSkill(ctx context.Context, request *getSkillRequest) (*getSkillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	skill, err := s.store.Skill(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownSkill
	}
	return &getSkillResponse{Body: skillOf(skill)}, nil
}

// updateSkill replaces a skill with what it now is.
func (s *Server) updateSkill(ctx context.Context, request *updateSkillRequest) (*updateSkillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if message, ok := skillComplaint(*request.Body); !ok {
		return nil, invalidRequest(message)
	}
	if _, err := s.configs.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return nil, errUnknownConfig
	}

	existing, err := s.store.Skill(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownSkill
	}

	skill := storedSkill(*request.Body, customerID)
	skill.ID = existing.ID
	skill.CreatedAt = existing.CreatedAt
	if err := s.configs.UpdateSkill(ctx, &skill); err != nil {
		return nil, storeFailure(err, errSkillNameTaken)
	}
	stored := skillOf(skill)
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditSkill, ResourceID: skill.ID, ResourceName: skill.Name,
		AgentID: skill.ConfigID, Action: store.AuditUpdated,
		Changes: auditDiff(skillOf(existing), stored),
	})
	return &updateSkillResponse{Body: stored}, nil
}

// deleteSkill stops a skill being usable.
func (s *Server) deleteSkill(ctx context.Context, request *deleteSkillRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}

	// Read before it goes, so the entry recording the deletion can say what was deleted and
	// which agent it was under.
	existing, err := s.store.Skill(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownSkill
	}
	if err := s.configs.DeleteSkill(ctx, customerID, request.Id); err != nil {
		return nil, errUnknownSkill
	}
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditSkill, ResourceID: existing.ID, ResourceName: existing.Name,
		AgentID: existing.ConfigID, Action: store.AuditDeleted,
		Changes: auditDiff(skillOf(existing), nil),
	})
	return nil, nil
}

// errUnknownConfig and errUnknownSkill are what a caller is told about a resource that is not
// theirs, which is the same thing they are told about one that never existed.
var (
	errUnknownConfig = APIError{
		Type: ErrorTypeNotFound, Code: codeAgentConfigNotFound,
		Message: "no such agent config",
	}
	errUnknownSkill = APIError{Type: ErrorTypeNotFound, Code: codeSkillNotFound, Message: "no such skill"}
)

// errAgentNameTaken and errSkillNameTaken are a create or a rename to a name that another
// live agent config of the customer, or another skill of the same config, already has.
var (
	errAgentNameTaken = APIError{
		Type: ErrorTypeConflict, Code: codeNameTaken,
		Message: "an agent with this name already exists",
	}
	errSkillNameTaken = APIError{
		Type: ErrorTypeConflict, Code: codeNameTaken,
		Message: "this agent already has a skill with this name",
	}
)

// storeFailure answers err from storing an agent config, a skill, a router config or a
// voice. A name that another live one of its kind has is taken, a 409 the caller fixes by
// choosing another name. A record or a connection that is not there is the invalid request
// it has always been answered with. Anything else is the database failing rather than the
// caller, so err goes back as it is, to be answered as a 500 and recorded with its stack.
func storeFailure(err error, taken APIError) error {
	switch {
	case errors.Is(err, store.ErrNameTaken):
		return taken
	case errors.Is(err, store.ErrNoAgentConfig), errors.Is(err, store.ErrNoSkill),
		errors.Is(err, store.ErrNoRouterConfig), errors.Is(err, store.ErrNoVoice),
		errors.Is(err, store.ErrNoConnectorConnection):
		return invalidRequest(err.Error())
	}
	return err
}

// configComplaint reports what is wrong with an agent config, if anything. Keyterms are
// checked here rather than left to the transcriber, because a list nobody can serve is
// worth hearing about while the config is being written and not once a call is running.
func configComplaint(request AgentConfigRequest) (string, bool) {
	if strings.TrimSpace(request.Name) == "" {
		return "an agent config needs a name", false
	}
	if _, ok := modeOf(request.Mode); !ok {
		return fmt.Sprintf("an agent is either %s or %s", store.AgentModeVoice, store.AgentModeText), false
	}
	if len(keytermsOf(request.Keyterms)) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}
	if _, ok := sandboxOf(request.Sandbox); !ok {
		return fmt.Sprintf("there is no sandbox provider called %q", *request.Sandbox), false
	}
	if complaint, ok := sandboxOptionsComplaint(request.SandboxOptions); !ok {
		return complaint, false
	}
	if _, ok := harnessOf(request.Harness); !ok {
		return fmt.Sprintf("there is no harness called %q", *request.Harness), false
	}
	if value(request.Speed) < 0 {
		return "a voice's speed cannot be negative", false
	}
	if complaint, ok := guardrailComplaint(request.Guardrail); !ok {
		return complaint, false
	}
	if complaint, ok := visibleToolsComplaint(request.VisibleTools); !ok {
		return complaint, false
	}
	if complaint, ok := connectorBindingsComplaint(request.Connectors); !ok {
		return complaint, false
	}
	return dispatchComplaint(request.Dispatch)
}

// aliasSeparator joins an alias to its tool's name in the name the model is offered,
// <alias>__<tool>, which is split back at the first one: Prefix and Split in
// internal/mcp/mcp.go:413-422 on codex/connector-support at cf62af0d, handed the alias by
// internal/session/connector_tools.go:167 there, as plugins.PrefixSeparator does for plugins
// today. An alias holding one would be split in the wrong place.
const aliasSeparator = "__"

// connectorBindingsComplaint reports what is wrong with a config's connector bindings as
// written that their schema cannot say, naming the binding. The alias pattern, the caps, the
// connection type, the timeout and the digest are tags on AgentConnectorBinding, which Huma
// checks first. Whether what they name exists is unboundConnectors', which asks the store.
// It is the prototype's connectorBindingsComplaint (internal/api/connectors.go:1147-1192 at
// cf62af0d).
func connectorBindingsComplaint(bindings *[]AgentConnectorBinding) (string, bool) {
	if bindings == nil {
		return "", true
	}
	aliases := make(map[string]bool, len(*bindings))
	for _, binding := range *bindings {
		alias := binding.Name
		if strings.Contains(alias, aliasSeparator) {
			return fmt.Sprintf("connector binding %q has %s in its name, which is what separates an alias "+
				"from its tool's name", alias, aliasSeparator), false
		}
		if aliases[alias] {
			return fmt.Sprintf("two connector bindings are called %q", alias), false
		}
		aliases[alias] = true
		if strings.TrimSpace(binding.ConnectorId) == "" {
			return fmt.Sprintf("connector binding %q names no connector_id", alias), false
		}
		switch binding.Connection.Type {
		case AgentConnectorSelectionTypeFixed:
			if strings.TrimSpace(value(binding.Connection.ConnectionId)) == "" {
				return fmt.Sprintf("connector binding %q is fixed, so it needs connection.connection_id", alias), false
			}
		case AgentConnectorSelectionTypeSession:
			if binding.Connection.ConnectionId != nil {
				return fmt.Sprintf("connector binding %q is chosen per session, so its connection is picked "+
					"when a session is created and connection.connection_id is not set here", alias), false
			}
		}
		if binding.Events != nil && len(*binding.Events) > 0 && binding.Connection.Type != AgentConnectorSelectionTypeFixed {
			return fmt.Sprintf("connector binding %q declares events, and only a fixed binding may: a session "+
				"binding's connection is picked when a session opens, and an event arrives with none open", alias), false
		}
		events := map[string]bool{}
		for _, event := range value(binding.Events) {
			if strings.TrimSpace(event.Event) == "" {
				return fmt.Sprintf("connector binding %q declares an event with no name", alias), false
			}
			key := mcpevents.Key(strings.TrimSpace(event.Event), value(event.Arguments))
			if events[key] {
				return fmt.Sprintf("connector binding %q declares event %q with the same arguments twice", alias, event.Event), false
			}
			events[key] = true
		}
		tools := make(map[string]bool, len(binding.Tools))
		for _, tool := range binding.Tools {
			if strings.TrimSpace(tool.Name) == "" {
				return fmt.Sprintf("connector binding %q grants a tool with no name", alias), false
			}
			if tools[tool.Name] {
				return fmt.Sprintf("connector binding %q grants %q twice", alias, tool.Name), false
			}
			if tool.SchemaDigest == "" && binding.Connection.Type != AgentConnectorSelectionTypeSession {
				return fmt.Sprintf("connector binding %q grants %q with no schema_digest, and only a session binding "+
					"may: a fixed binding's connection is the app's own, so grant the digest GET "+
					"/v1/agents/connections/{id}/tools lists for it", alias, tool.Name), false
			}
			tools[tool.Name] = true
		}
	}
	return "", true
}

// unboundConnectors reports a binding naming what the app cannot bind: a connector it cannot
// see, built-in or its own, or for a fixed binding anything but a live connection the app
// itself owns. A user's connection is theirs to use in their own sessions, and a fixed
// binding would hand it to every session the config runs. An error is the store failing,
// not the binding.
func (s *Server) unboundConnectors(ctx context.Context, customerID string, bindings *[]AgentConnectorBinding) (string, bool, error) {
	if bindings == nil {
		return "", true, nil
	}
	for _, binding := range *bindings {
		_, err := s.store.LatestConnectorDefinition(ctx, customerID, binding.ConnectorId)
		if errors.Is(err, store.ErrNoConnectorDefinition) {
			return fmt.Sprintf("connector binding %q names connector %q, and there is no such connector",
				binding.Name, binding.ConnectorId), false, nil
		}
		if err != nil {
			return "", false, err
		}
		if binding.Connection.Type != AgentConnectorSelectionTypeFixed {
			continue
		}
		id := value(binding.Connection.ConnectionId)
		connection, err := s.store.ConnectorConnection(ctx, customerID, id)
		if errors.Is(err, store.ErrNoConnectorConnection) {
			return fmt.Sprintf("connector binding %q names connection %q, and the app has no such connection",
				binding.Name, id), false, nil
		}
		if err != nil {
			return "", false, err
		}
		if connection.OwnerType != store.OwnerApp {
			return fmt.Sprintf("connector binding %q is fixed, so its connection has to be the app's own, and "+
				"%q is a user's: bind it with connection.type session instead", binding.Name, id), false, nil
		}
		// The grants and digests describe the connector the binding names, so a connection to
		// another one would call them with the wrong provider's credentials.
		if connection.ConnectorID != binding.ConnectorId {
			return fmt.Sprintf("connector binding %q names connector %q, but connection %q is to %q",
				binding.Name, binding.ConnectorId, id, connection.ConnectorID), false, nil
		}
	}
	return "", true, nil
}

// pluginAliasComplaint reports a binding called what a plugin or an MCP server of the same
// config is. A plugin's tools are offered as <plugin>__<tool> (plugins.Prefix,
// internal/plugins/mcp.go), a user plugin's as <plugin>__ and a suffix
// (internal/plugins/user.go), an MCP server's as <name>__<tool> (McpServer.Name), and a
// binding's as <alias>__<tool>. The built-in connectors share ids with the plugin catalog
// (slack is in both internal/plugins/plugins.yaml and
// internal/connectors/providers/slack.yaml), so the two would offer the same names. It reads
// the config as it is about to be stored, so a patch or a sync adding either side is checked
// against what the other already is.
//
// A binding to the plugin's own connector may be called what the plugin is: the session drops
// a plugin entry whose connector a binding names (session.Spec.withoutBoundPlugins), so only
// the binding offers the name. That is the binding router plugins migrate writes beside the
// entry it keeps (internal/pluginmigrate), and the config stays editable (AI-994 F42).
func pluginAliasComplaint(config store.AgentConfig) (string, bool) {
	for _, binding := range config.Connectors {
		if binding.ConnectorID != binding.Name &&
			(namesPluginEntry(config.AgentPlugins, binding.Name) || namesPluginEntry(config.UserPlugins, binding.Name)) {
			return fmt.Sprintf("connector binding %q is called what the config's plugin %q is, and both "+
				"would offer their tools as %s%stool", binding.Name, binding.Name, binding.Name, aliasSeparator), false
		}
		for _, server := range config.MCPServers {
			if server.Name == binding.Name {
				return fmt.Sprintf("connector binding %q is called what the config's MCP server %q is, and both "+
					"would offer their tools as %s%stool", binding.Name, binding.Name, binding.Name, aliasSeparator), false
			}
		}
	}
	return "", true
}

// namesPluginEntry reports whether any of the entries is called name.
func namesPluginEntry(entries []store.PluginEntry, name string) bool {
	for _, entry := range entries {
		if entry.Name == name {
			return true
		}
	}
	return false
}

// storedBindings turns the bindings a caller sent into what a config stores, as written.
func storedBindings(bindings []AgentConnectorBinding) []store.ConnectorBinding {
	stored := make([]store.ConnectorBinding, 0, len(bindings))
	for _, binding := range bindings {
		tools := make([]store.ToolGrant, 0, len(binding.Tools))
		for _, tool := range binding.Tools {
			tools = append(tools, store.ToolGrant{Name: tool.Name, SchemaDigest: tool.SchemaDigest})
		}
		stored = append(stored, store.ConnectorBinding{
			Name:        binding.Name,
			ConnectorID: binding.ConnectorId,
			Connection: store.ConnectionBinding{
				Type:         string(binding.Connection.Type),
				ConnectionID: value(binding.Connection.ConnectionId),
			},
			Tools:     tools,
			Required:  value(binding.Required),
			TimeoutMs: value(binding.TimeoutMs),
			Events:    bindingEventsOf(binding.Events),
			Policy:    bindingPolicyOf(binding.Policy),
		})
	}
	return stored
}

// bindingPolicyOf reads the policy a caller wrote on a binding, or nil for none, so a binding
// without one is stored as it was before policies existed.
func bindingPolicyOf(policy *ConnectorBindingPolicy) *store.BindingPolicy {
	if policy == nil {
		return nil
	}
	return &store.BindingPolicy{
		PreSpeech:   value(policy.PreSpeech),
		OnInterrupt: string(value(policy.OnInterrupt)),
		Cancellable: policy.Cancellable,
	}
}

// bindingEventsOf reads the events a caller declared on a binding, or nothing for none, so a
// binding without events is stored as it was before they existed.
func bindingEventsOf(events *[]ConnectorBindingEvent) []store.BindingEvent {
	var read []store.BindingEvent
	for _, event := range value(events) {
		read = append(read, store.BindingEvent{
			Event:        strings.TrimSpace(event.Event),
			Arguments:    value(event.Arguments),
			Instructions: strings.TrimSpace(value(event.Instructions)),
		})
	}
	return read
}

// bindingsOf renders a config's bindings for the wire.
func bindingsOf(bindings []store.ConnectorBinding) []AgentConnectorBinding {
	rendered := make([]AgentConnectorBinding, 0, len(bindings))
	for _, binding := range bindings {
		tools := make([]ConnectorToolGrant, 0, len(binding.Tools))
		for _, tool := range binding.Tools {
			tools = append(tools, ConnectorToolGrant{Name: tool.Name, SchemaDigest: tool.SchemaDigest})
		}
		required := binding.Required
		one := AgentConnectorBinding{
			Name:        binding.Name,
			ConnectorId: binding.ConnectorID,
			Connection: AgentConnectorSelection{
				Type:         AgentConnectorSelectionType(binding.Connection.Type),
				ConnectionId: optional(binding.Connection.ConnectionID),
			},
			Tools:    tools,
			Required: &required,
		}
		if binding.TimeoutMs > 0 {
			timeout := binding.TimeoutMs
			one.TimeoutMs = &timeout
		}
		if len(binding.Events) > 0 {
			events := make([]ConnectorBindingEvent, 0, len(binding.Events))
			for _, event := range binding.Events {
				declared := ConnectorBindingEvent{Event: event.Event}
				if len(event.Arguments) > 0 {
					declared.Arguments = &event.Arguments
				}
				if event.Instructions != "" {
					declared.Instructions = &event.Instructions
				}
				events = append(events, declared)
			}
			one.Events = &events
		}
		if policy := binding.Policy; policy != nil {
			one.Policy = &ConnectorBindingPolicy{PreSpeech: optional(policy.PreSpeech), Cancellable: policy.Cancellable}
			if policy.OnInterrupt != "" {
				onInterrupt := ConnectorOnInterrupt(policy.OnInterrupt)
				one.Policy.OnInterrupt = &onInterrupt
			}
		}
		rendered = append(rendered, one)
	}
	return rendered
}

// textThinkingComplaint refuses a thinking model on a text agent, which runs everything on
// its llm, and drops the one a voice agent kept when it became a text one. It reads the
// config as it will be stored, since a patch or a sync can change the mode without naming
// a thinking model.
func textThinkingComplaint(config *store.AgentConfig, asked *string) (string, bool) {
	if config.Mode != store.AgentModeText {
		return "", true
	}
	if value(asked) != "" {
		return "thinking_llm is only for voice agents: a text agent runs everything, skills included, on its llm", false
	}
	config.Subagent = ""
	return "", true
}

// dispatchComplaint reports a dispatch setting that is neither enabled nor disabled.
func dispatchComplaint(asked *AgentDispatch) (string, bool) {
	if asked == nil {
		return "", true
	}
	if asked.IncomingCall != nil && !asked.IncomingCall.Valid() {
		return fmt.Sprintf("dispatch.incoming_call is %s or %s, not %q", Enabled, Disabled, *asked.IncomingCall), false
	}
	if asked.Text != nil && !asked.Text.Valid() {
		return fmt.Sprintf("dispatch.text is %s or %s, not %q", Enabled, Disabled, *asked.Text), false
	}
	return "", true
}

// applyDispatch writes what a caller said about dispatch onto a config. A setting left out
// keeps what is stored.
func applyDispatch(config *store.AgentConfig, asked *AgentDispatch) {
	if asked == nil {
		return
	}
	if asked.IncomingCall != nil {
		config.DispatchIncomingCall = *asked.IncomingCall == Enabled
	}
	if asked.Text != nil {
		config.DispatchText = *asked.Text == Enabled
	}
}

// dispatchOf renders what a config leaves to dispatch workers, spelling out both settings.
func dispatchOf(config store.AgentConfig) *AgentDispatch {
	setting := func(enabled bool) *DispatchSetting {
		rendered := Disabled
		if enabled {
			rendered = Enabled
		}
		return &rendered
	}
	return &AgentDispatch{
		IncomingCall: setting(config.DispatchIncomingCall),
		Text:         setting(config.DispatchText),
	}
}

// visibleToolsComplaint reports what is wrong with the tools a config shows end users, if
// anything. A pattern that cannot match is refused here rather than found showing nothing.
func visibleToolsComplaint(patterns *[]string) (string, bool) {
	if patterns == nil {
		return "", true
	}
	if len(*patterns) > 64 {
		return "a config may show at most 64 visible_tools", false
	}
	for _, pattern := range *patterns {
		if !conversation.ValidVisibleTool(pattern) {
			return fmt.Sprintf("visible_tools has %q, which is not a tool name or pattern", pattern), false
		}
	}
	return "", true
}

// guardrailComplaint reports what is wrong with a guardrail policy, if anything.
//
// It is read here, as it is written, rather than when a call starts: a policy that will
// not parse is a config that screens nothing, and finding that out at the first turn means
// finding it out from an agent that answered a question it was meant to refuse.
func guardrailComplaint(policy *string) (string, bool) {
	if policy == nil || strings.TrimSpace(*policy) == "" {
		return "", true
	}
	if _, err := guardrail.Parse(*policy); err != nil {
		return err.Error(), false
	}
	return "", true
}

// modeOf reads the mode a caller sent, which is optional and defaults to voice. An
// unknown one is refused rather than defaulted, since a text agent asked for as "txt"
// would otherwise quietly join a call.
func modeOf(mode *AgentMode) (string, bool) {
	if mode == nil || *mode == "" {
		return store.AgentModeVoice, true
	}
	switch string(*mode) {
	case store.AgentModeVoice, store.AgentModeText:
		return string(*mode), true
	}
	return "", false
}

// sandboxOptionsComplaint reports what is wrong with how a caller asked for the sandbox to
// be built, if anything. The bounds are the schema's, held here as well because a config
// written through sync or a patch never meets the generated validator.
func sandboxOptionsComplaint(options *SandboxOptions) (string, bool) {
	if options == nil {
		return "", true
	}
	bounded := []struct {
		name       string
		value, max int
	}{
		{"timeout_ms", value(options.TimeoutMs), int(sandbox.MaxTimeout.Milliseconds())},
		{"cpu", value(options.Cpu), 16},
		{"memory_gb", value(options.MemoryGb), 64},
		{"disk_gb", value(options.DiskGb), 100},
	}
	for _, field := range bounded {
		if field.value < 0 || field.value > field.max {
			return fmt.Sprintf("sandbox_options.%s is between 0 and %d", field.name, field.max), false
		}
	}
	if len(value(options.Image)) > 256 {
		return "sandbox_options.image is at most 256 characters", false
	}
	if len(value(options.Setup)) > 32 {
		return "sandbox_options.setup is at most 32 commands", false
	}
	for _, command := range value(options.Setup) {
		if strings.TrimSpace(command) == "" || len(command) > 2048 {
			return "a sandbox_options.setup command is between 1 and 2048 characters", false
		}
	}
	return "", true
}

// sandboxConfigOf reads how a caller asked for the sandbox to be built.
func sandboxConfigOf(options *SandboxOptions) sandbox.Config {
	if options == nil {
		return sandbox.Config{}
	}
	return sandbox.Config{
		Image:     strings.TrimSpace(value(options.Image)),
		Setup:     value(options.Setup),
		TimeoutMs: value(options.TimeoutMs),
		CPU:       value(options.Cpu),
		MemoryGB:  value(options.MemoryGb),
		DiskGB:    value(options.DiskGb),
	}
}

// sandboxOptionsOf renders how a config's sandbox is built, or nothing when it is the
// provider's own.
func sandboxOptionsOf(config sandbox.Config) *SandboxOptions {
	if config.Image == "" && len(config.Setup) == 0 && config.TimeoutMs == 0 &&
		config.CPU == 0 && config.MemoryGB == 0 && config.DiskGB == 0 {
		return nil
	}
	setup := append([]string{}, config.Setup...)
	return &SandboxOptions{
		Image:     optional(config.Image),
		Setup:     &setup,
		TimeoutMs: &config.TimeoutMs,
		Cpu:       &config.CPU,
		MemoryGb:  &config.MemoryGB,
		DiskGb:    &config.DiskGB,
	}
}

// pluginEventsComplaint reports what is wrong with the events a config subscribes to, if
// anything. It reads the config as it will be stored, since whether an event's plugin is
// named depends on plugins and user_plugins as well.
func pluginEventsComplaint(config store.AgentConfig) (string, bool) {
	for _, event := range config.PluginEvents {
		if _, ok := plugins.Lookup(event.Plugin); !ok {
			return fmt.Sprintf("plugin_events: no plugin called %q", event.Plugin), false
		}
		if !store.NamesPlugin(config.AgentPlugins, event.Plugin) && !store.NamesPlugin(config.UserPlugins, event.Plugin) {
			return fmt.Sprintf("plugin_events: %s is named under neither agent_plugins nor user_plugins", event.Plugin), false
		}
		if event.Event == "" {
			return fmt.Sprintf("plugin_events: a %s event needs a name", event.Plugin), false
		}
	}
	return "", true
}

// pluginEntriesComplaint reports what is wrong with the plugins a config names, if anything:
// an id the catalog does not have, one named twice in a list, or options its catalog entry
// does not allow.
func pluginEntriesComplaint(config store.AgentConfig) (string, bool) {
	for _, list := range []struct {
		field   string
		entries []store.PluginEntry
	}{{"agent_plugins", config.AgentPlugins}, {"user_plugins", config.UserPlugins}} {
		field, entries := list.field, list.entries
		seen := map[string]bool{}
		for _, entry := range entries {
			plugin, ok := plugins.Lookup(entry.Name)
			if !ok {
				return fmt.Sprintf("%s: no plugin called %q", field, entry.Name), false
			}
			if seen[entry.Name] {
				return fmt.Sprintf("%s: %s is named twice", field, entry.Name), false
			}
			seen[entry.Name] = true
			if _, err := plugin.Configured(plugins.Options{
				Readonly: entry.Readonly, Scopes: entry.Scopes, Toolsets: entry.Toolsets, Tools: entry.Tools,
			}); err != nil {
				return field + ": " + strings.TrimPrefix(err.Error(), "plugins: "), false
			}
		}
	}
	return "", true
}

// pluginEntriesOf reads the plugins a caller named, with blank scopes and toolsets left out.
func pluginEntriesOf(entries []PluginEntry) []store.PluginEntry {
	read := []store.PluginEntry{}
	for _, entry := range entries {
		read = append(read, store.PluginEntry{
			Name:     strings.TrimSpace(entry.Name),
			Readonly: value(entry.Readonly),
			Scopes:   nonBlank(value(entry.Scopes)),
			Toolsets: nonBlank(value(entry.Toolsets)),
			Tools:    nonBlank(value(entry.Tools)),
		})
	}
	return read
}

func nonBlank(values []string) []string {
	var kept []string
	for _, item := range values {
		if item = strings.TrimSpace(item); item != "" {
			kept = append(kept, item)
		}
	}
	return kept
}

// renderedPluginEntries is the plugins a config names as the API shows them: a bare id for
// one with no options, or nothing for none.
func renderedPluginEntries(entries []store.PluginEntry) *[]PluginEntry {
	if len(entries) == 0 {
		return nil
	}
	rendered := make([]PluginEntry, 0, len(entries))
	for _, entry := range entries {
		shown := PluginEntry{Name: entry.Name}
		if entry.Readonly {
			readonly := true
			shown.Readonly = &readonly
		}
		if len(entry.Scopes) > 0 {
			scopes := entry.Scopes
			shown.Scopes = &scopes
		}
		if len(entry.Toolsets) > 0 {
			toolsets := entry.Toolsets
			shown.Toolsets = &toolsets
		}
		shown.Tools = shownTools(entry.Tools)
		rendered = append(rendered, shown)
	}
	return &rendered
}

// mcpServersComplaint reports what is wrong with the MCP servers a config names, if
// anything. A name prefixes the server's tools the way a plugin's id does, so it may not be
// one, nor hold the separator that splits a tool's name from it.
func mcpServersComplaint(config store.AgentConfig) (string, bool) {
	seen := map[string]bool{}
	for _, server := range config.MCPServers {
		if strings.Contains(server.Name, plugins.PrefixSeparator) {
			return fmt.Sprintf("mcp_servers: %s may not contain %s", server.Name, plugins.PrefixSeparator), false
		}
		if _, ok := plugins.Lookup(server.Name); ok {
			return fmt.Sprintf("mcp_servers: %s is a catalog plugin; name it under agent_plugins or user_plugins instead", server.Name), false
		}
		if seen[server.Name] {
			return fmt.Sprintf("mcp_servers: %s is named twice", server.Name), false
		}
		seen[server.Name] = true
		endpoint, err := url.Parse(server.URL)
		if err != nil || endpoint.Scheme != "https" || endpoint.Host == "" {
			return fmt.Sprintf("mcp_servers: %s needs an https url", server.Name), false
		}
		if err := plugins.CheckToolPatterns(server.Tools); err != nil {
			return fmt.Sprintf("mcp_servers: %s: %s", server.Name, strings.TrimPrefix(err.Error(), "plugins: ")), false
		}
	}
	return "", true
}

// mcpServersOf reads the MCP servers a caller named.
func mcpServersOf(servers *[]McpServer) []store.MCPServer {
	read := []store.MCPServer{}
	for _, server := range value(servers) {
		read = append(read, store.MCPServer{
			Name:   strings.TrimSpace(server.Name),
			URL:    strings.TrimSpace(server.Url),
			Tools:  nonBlank(value(server.Tools)),
			Scopes: nonBlank(value(server.Scopes)),
			User:   value(server.User),
		})
	}
	return read
}

// shownTools is a tool allowlist as the API shows it, or nothing for every tool.
func shownTools(tools []string) *[]string {
	if len(tools) == 0 {
		return nil
	}
	shown := tools
	return &shown
}

// renderedMCPServers is a config's MCP servers as the API shows them, or nothing for none.
func renderedMCPServers(servers []store.MCPServer) *[]McpServer {
	if len(servers) == 0 {
		return nil
	}
	rendered := make([]McpServer, 0, len(servers))
	for _, server := range servers {
		shown := McpServer{Name: server.Name, Url: server.URL, Tools: shownTools(server.Tools)}
		if len(server.Scopes) > 0 {
			scopes := server.Scopes
			shown.Scopes = &scopes
		}
		if server.User {
			user := true
			shown.User = &user
		}
		shown.NeedsLogin = server.NeedsLogin
		if branding := server.Branding; branding != nil {
			shown.Branding = &McpServerBranding{
				Title:       optional(branding.Title),
				Description: optional(branding.Description),
				Version:     optional(branding.Version),
				IconUrl:     optional(branding.IconURL),
				WebsiteUrl:  optional(branding.WebsiteURL),
			}
		}
		rendered = append(rendered, shown)
	}
	return &rendered
}

// describedMCPServers is servers with each one not asked yet asked how it describes itself
// and whether it needs a login, or what is wrong with one whose login cannot be made as the
// config says. A server that does not answer in time keeps what it said before at the same
// URL, or goes without, and the config is saved either way.
func (s *Server) describedMCPServers(ctx context.Context, servers, before []store.MCPServer) ([]store.MCPServer, string, bool) {
	described := slices.Clone(servers)
	complaints := make([]string, len(described))
	ctx, cancel := context.WithTimeout(ctx, mcpBrandingTimeout)
	defer cancel()
	var wg sync.WaitGroup
	for i := range described {
		server := &described[i]
		if server.Branding == nil {
			wg.Go(func() { s.brand(ctx, server, before) })
		}
		if server.NeedsLogin == nil {
			wg.Go(func() { complaints[i] = s.askLogin(ctx, server, before) })
		}
	}
	wg.Wait()
	for _, complaint := range complaints {
		if complaint != "" {
			return nil, complaint, false
		}
	}
	return described, "", true
}

// brand asks the server how it describes itself.
func (s *Server) brand(ctx context.Context, server *store.MCPServer, before []store.MCPServer) {
	described, err := plugins.Describe(ctx, plugins.Connection{PluginID: server.Name, Endpoint: server.URL}, s.auth().HTTP)
	if err != nil {
		s.logger.Debug("mcp server did not describe itself", "server", server.Name, "error", err)
		server.Branding = earlierServer(before, server.URL).Branding
		return
	}
	if described != (plugins.Branding{}) {
		server.Branding = &store.MCPBranding{
			Title: described.Title, Description: described.Description, Version: described.Version,
			IconURL: described.IconURL, WebsiteURL: described.WebsiteURL,
		}
	}
}

// askLogin records whether the server needs a login, and reports what is wrong if its
// login cannot be made or the config asks for one it does not have.
func (s *Server) askLogin(ctx context.Context, server *store.MCPServer, before []store.MCPServer) string {
	needs, err := s.auth().NeedsLogin(ctx, server.URL)
	if err != nil {
		s.logger.Debug("mcp server could not be asked whether it needs a login", "server", server.Name, "error", err)
		server.NeedsLogin = earlierServer(before, server.URL).NeedsLogin
		return ""
	}
	server.NeedsLogin = &needs
	if !needs {
		if server.User || len(server.Scopes) > 0 {
			return fmt.Sprintf("mcp_servers: %s sets scopes or user, but advertises no OAuth login", server.Name)
		}
		return ""
	}
	var unreachable *url.Error
	if err := s.auth().CheckLogin(ctx, server.URL); err != nil && !errors.As(err, &unreachable) {
		return fmt.Sprintf("mcp_servers: %s needs an OAuth login the router cannot make: %s",
			server.Name, strings.TrimPrefix(err.Error(), "plugins: "))
	}
	return ""
}

// earlierServer is the server that was at endpoint before, or none.
func earlierServer(before []store.MCPServer, endpoint string) store.MCPServer {
	for _, server := range before {
		if server.URL == endpoint {
			return server
		}
	}
	return store.MCPServer{}
}

// pluginEventsOf reads the events a caller subscribed the config to.
func pluginEventsOf(events *[]PluginEvent) []store.PluginEvent {
	read := []store.PluginEvent{}
	for _, event := range value(events) {
		read = append(read, store.PluginEvent{
			Plugin:       strings.TrimSpace(event.Plugin),
			Event:        strings.TrimSpace(event.Event),
			Arguments:    value(event.Arguments),
			Instructions: strings.TrimSpace(value(event.Instructions)),
		})
	}
	return read
}

// renderedPluginEvents is a config's events as the API shows them, or nothing for none.
func renderedPluginEvents(events []store.PluginEvent) *[]PluginEvent {
	if len(events) == 0 {
		return nil
	}
	rendered := make([]PluginEvent, 0, len(events))
	for _, event := range events {
		shown := PluginEvent{Plugin: event.Plugin, Event: event.Event, Instructions: optional(event.Instructions)}
		if len(event.Arguments) > 0 {
			arguments := event.Arguments
			shown.Arguments = &arguments
		}
		rendered = append(rendered, shown)
	}
	return &rendered
}

// sandboxOf reads the sandbox a caller sent, which is optional. An unknown one is refused
// rather than dropped, since a config that quietly runs no code is hard to tell from one
// whose subagent simply chose not to.
func sandboxOf(box *Sandbox) (string, bool) {
	if box == nil || *box == "" {
		return "", true
	}
	if !box.Valid() {
		return "", false
	}
	return string(*box), true
}

// harnessOf reads the harness a caller sent, which is optional and defaults to the default
// one. An unknown one is refused rather than defaulted, since an agent asked to run one that
// does not exist would otherwise quietly run another.
func harnessOf(named *Harness) (string, bool) {
	if named == nil || *named == "" {
		return harness.Default, true
	}
	if !named.Valid() {
		return "", false
	}
	return string(*named), true
}

// keytermsOf reads the terms a caller sent, which are optional and may be blank.
func keytermsOf(list *[]string) []string {
	if list == nil {
		return nil
	}
	return stt.CleanKeyterms(*list)
}

// skillComplaint reports what is wrong with a skill, if anything. A skill without a
// description is one the fast model would never know when to reach for.
func skillComplaint(request SkillRequest) (string, bool) {
	if strings.TrimSpace(request.ConfigId) == "" {
		return "a skill belongs to an agent config, so one has to be named", false
	}
	if strings.TrimSpace(request.Name) == "" {
		return "a skill needs a name", false
	}
	if strings.TrimSpace(request.Description) == "" {
		return "a skill needs a description, which is how the model decides when to use it", false
	}
	if strings.TrimSpace(request.Instructions) == "" {
		return "a skill needs instructions, which is what the subagent answers under", false
	}
	return "", true
}

// storedConfig turns a request into a row. The customer comes from the trusted header
// rather than the body, the same way a session's does.
func storedConfig(request AgentConfigRequest, customerID string) store.AgentConfig {
	mode, _ := modeOf(request.Mode)
	box, _ := sandboxOf(request.Sandbox)
	named, _ := harnessOf(request.Harness)
	config := store.AgentConfig{
		CustomerID:         customerID,
		Name:               strings.TrimSpace(request.Name),
		Mode:               mode,
		STT:                value(request.Stt),
		TTS:                value(request.Tts),
		STS:                value(request.Sts),
		Voice:              value(request.Voice),
		Speed:              value(request.Speed),
		LLM:                value(request.Llm),
		Subagent:           value(request.ThinkingLlm),
		Search:             value(request.Search),
		Instructions:       value(request.Instructions),
		Greeting:           value(request.Greeting),
		Guardrail:          value(request.Guardrail),
		KnowledgeNamespace: value(request.KnowledgeNamespace),
		Sandbox:            box,
		SandboxOptions:     sandboxConfigOf(request.SandboxOptions),
		Harness:            named,
	}
	if request.Skills != nil {
		config.Skills = *request.Skills
	}
	if request.AgentPlugins != nil {
		config.AgentPlugins = pluginEntriesOf(*request.AgentPlugins)
	}
	if request.Connectors != nil {
		config.Connectors = storedBindings(*request.Connectors)
	}
	if request.UserPlugins != nil {
		config.UserPlugins = pluginEntriesOf(*request.UserPlugins)
	}
	config.PluginEvents = pluginEventsOf(request.PluginEvents)
	config.MCPServers = mcpServersOf(request.McpServers)
	config.Channels = channelsOf(request.Channels)
	config.Keyterms = keytermsOf(request.Keyterms)
	if request.VisibleTools != nil {
		config.VisibleTools = *request.VisibleTools
	}
	if request.Tags != nil {
		config.Tags = *request.Tags
	}
	if request.Video != nil {
		config.VideoSource = value(request.Video.Source)
		config.VideoMaxFrames = value(request.Video.MaxFrames)
	}
	applyDispatch(&config, request.Dispatch)
	config.EpisodeCards = value(request.EpisodeCards)
	config.ProgressiveTools = value(request.ProgressiveTools)
	return config
}

// storedSkill turns a request into a row.
func storedSkill(request SkillRequest, customerID string) store.Skill {
	return store.Skill{
		CustomerID:   customerID,
		ConfigID:     strings.TrimSpace(request.ConfigId),
		Name:         strings.TrimSpace(request.Name),
		Description:  request.Description,
		Instructions: request.Instructions,
		DeadlineMs:   value(request.DeadlineMs),
		CaptureVideo: value(request.CaptureVideo),
	}
}

// agentConfigOf renders a config for the wire. Empty strings are left out rather than
// sent as blanks, so a config that says nothing about a model reads as saying nothing.
func agentConfigOf(config store.AgentConfig) AgentConfig {
	rendered := AgentConfig{
		Id:        config.ID,
		Name:      config.Name,
		Mode:      AgentMode(config.Mode),
		CreatedAt: config.CreatedAt,
		UpdatedAt: config.UpdatedAt,
	}
	rendered.Stt = optional(config.STT)
	rendered.Tts = optional(config.TTS)
	rendered.Sts = optional(config.STS)
	rendered.Voice = optional(config.Voice)
	if config.Speed != 0 {
		speed := config.Speed
		rendered.Speed = &speed
	}
	rendered.Llm = optional(config.LLM)
	rendered.ThinkingLlm = optional(config.Subagent)
	frames := config.VideoMaxFrames
	if frames == 0 {
		frames = 1
	}
	rendered.Video = &SessionVideo{Source: optional(config.VideoSource), MaxFrames: &frames}
	rendered.Search = optional(config.Search)
	rendered.Instructions = optional(config.Instructions)
	rendered.Greeting = optional(config.Greeting)
	rendered.Guardrail = optional(config.Guardrail)
	rendered.KnowledgeNamespace = optional(config.KnowledgeNamespace)
	if config.Sandbox != "" {
		box := Sandbox(config.Sandbox)
		rendered.Sandbox = &box
	}
	rendered.SandboxOptions = sandboxOptionsOf(config.SandboxOptions)
	named := Harness(config.Harness)
	if named == "" {
		named = Default
	}
	rendered.Harness = &named
	rendered.Dispatch = dispatchOf(config)
	episodeCards := config.EpisodeCards
	rendered.EpisodeCards = &episodeCards
	progressiveTools := config.ProgressiveTools
	rendered.ProgressiveTools = &progressiveTools
	if len(config.Skills) > 0 {
		skills := config.Skills
		rendered.Skills = &skills
	}
	rendered.AgentPlugins = renderedPluginEntries(config.AgentPlugins)
	rendered.UserPlugins = renderedPluginEntries(config.UserPlugins)
	if len(config.Connectors) > 0 {
		bindings := bindingsOf(config.Connectors)
		rendered.Connectors = &bindings
	}
	rendered.PluginEvents = renderedPluginEvents(config.PluginEvents)
	rendered.McpServers = renderedMCPServers(config.MCPServers)
	rendered.Channels = renderedChannels(config.Channels)
	if len(config.Keyterms) > 0 {
		keyterms := config.Keyterms
		rendered.Keyterms = &keyterms
	}
	if len(config.VisibleTools) > 0 {
		visible := config.VisibleTools
		rendered.VisibleTools = &visible
	}
	if len(config.Tags) > 0 {
		tags := config.Tags
		rendered.Tags = &tags
	}
	rendered.SyncHash = optional(config.SyncHash)
	return rendered
}

// skillOf renders a skill for the wire.
func skillOf(skill store.Skill) Skill {
	rendered := Skill{
		Id:           skill.ID,
		ConfigId:     skill.ConfigID,
		Name:         skill.Name,
		Description:  skill.Description,
		Instructions: skill.Instructions,
		CreatedAt:    skill.CreatedAt,
		UpdatedAt:    skill.UpdatedAt,
	}
	rendered.CaptureVideo = &skill.CaptureVideo
	if skill.DeadlineMs > 0 {
		deadline := skill.DeadlineMs
		rendered.DeadlineMs = &deadline
	}
	return rendered
}

// optional carries a string only when there is one, which is how an unset field stays
// unset on the way back out.
func optional(text string) *string {
	if text == "" {
		return nil
	}
	return &text
}

// registerConfigs declares the operations served in configs.go.
func (s *Server) registerConfigs(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listAgentConfigs",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs",
		Summary:     "The agent configs the calling customer holds",
		Description: "With a name this is how a name becomes a config, which is what lets a backend say " +
			"\"docs\" instead of an id it never chose.\n" +
			"Server-side only, as it always was: a config carries the instructions the agent runs " +
			"under, and those are not a page's business. A page addressing an agent by name does not " +
			"need this -- it sends the name on the create-session request and the router resolves " +
			"it, which is the same lookup without handing the instructions over.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's configs, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listAgentConfigs)
	huma.Register(api, huma.Operation{
		OperationID: "createAgentConfig",
		Method:      http.MethodPost,
		Path:        "/v1/agents/configs",
		Summary:     "Store a named configuration a session can be created from",
		Description: "A config holds what a caller would otherwise repeat on every call: the models, the " +
			"voice, the instructions and which skills the subagent may be handed. What is about one " +
			"conversation rather than the agent behind it, the call id above all, stays in the " +
			"create-session request.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The config was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict},
	}, s.createAgentConfig)
	huma.Register(api, huma.Operation{
		OperationID: "getAgentConfig",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs/{id}",
		Summary:     "One agent config",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getAgentConfig)
	huma.Register(api, huma.Operation{
		OperationID: "updateAgentConfig",
		Method:      http.MethodPut,
		Path:        "/v1/agents/configs/{id}",
		Summary:     "Replace an agent config",
		Description: "Every field is written, so the body is what the config now is rather than what changed " +
			"about it. Sessions already running keep the configuration they started with.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.updateAgentConfig)
	huma.Register(api, huma.Operation{
		OperationID: "deleteAgentConfig",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/configs/{id}",
		Summary:     "Delete an agent config",
		Description: "Calls that already ran under it keep naming it, so the config stops being usable rather " +
			"than stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The config is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteAgentConfig)
	huma.Register(api, huma.Operation{
		OperationID: "listSkills",
		Method:      http.MethodGet,
		Path:        "/v1/agents/skills",
		Summary:     "The skills the calling customer has defined",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's skills, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listSkills)
	huma.Register(api, huma.Operation{
		OperationID: "createSkill",
		Method:      http.MethodPost,
		Path:        "/v1/agents/skills",
		Summary:     "Define a kind of work worth handing to the slower model",
		Description: "A skill belongs to one agent config. Two agents that both need the same kind of work " +
			"have one each, so editing what \"explain\" means for one leaves the other alone. The " +
			"built-in think, recall and explain need no row: a config may name them without defining " +
			"them.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The skill was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.createSkill)
	huma.Register(api, huma.Operation{
		OperationID: "getSkill",
		Method:      http.MethodGet,
		Path:        "/v1/agents/skills/{id}",
		Summary:     "One skill",
		Responses: map[string]*huma.Response{
			"200": {Description: "The skill"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getSkill)
	huma.Register(api, huma.Operation{
		OperationID: "updateSkill",
		Method:      http.MethodPut,
		Path:        "/v1/agents/skills/{id}",
		Summary:     "Replace a skill",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The skill as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.updateSkill)
	huma.Register(api, huma.Operation{
		OperationID: "deleteSkill",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/skills/{id}",
		Summary:     "Delete a skill",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		DefaultStatus: http.StatusNoContent,
		Responses: map[string]*huma.Response{
			"204": {Description: "The skill is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteSkill)
}

type listAgentConfigsRequest struct {
	Name optionalParam[string] `query:"name" doc:"Narrow the list to the config with this name, which is how a name is resolved to a config. Names are unique per customer, so this answers with at most one."`
}

type listAgentConfigsResponse struct {
	Body []AgentConfig `nullable:"false"`
}

type createAgentConfigRequest struct {
	Body *AgentConfigRequest `required:"true"`
}

type createAgentConfigResponse struct {
	Body AgentConfig
}

type getAgentConfigRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getAgentConfigResponse struct {
	Body AgentConfig
}

type updateAgentConfigRequest struct {
	Id   string              `path:"id" doc:"The resource, as returned when it was created."`
	Body *AgentConfigRequest `required:"true"`
}

type updateAgentConfigResponse struct {
	Body AgentConfig
}

type deleteAgentConfigRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

// AgentConfigRequest is the AgentConfigRequest schema.
type AgentConfigRequest struct {
	Dispatch           *AgentDispatch           `json:"dispatch,omitempty"`
	EpisodeCards       *bool                    `json:"episode_cards,omitempty" doc:"Whether each phone call under this agent writes an episode card into the caller's omni-channel: an agent channel for each caller number and agent, keyed by the caller's E.164 number. On, a session on a thread channel or a phone call under this agent also starts with the person's other episode cards: a summary, or the last lines of the episode's channel while there is none. Off by default, and then a session runs as it always did. Left out on an update, the stored setting stays."`
	Greeting           *string                  `json:"greeting,omitempty"`
	Guardrail          *string                  `json:"guardrail,omitempty" doc:"A guardrail.md: frontmatter saying how a turn is screened - lcm, webhook or llm - then the policy in prose. A turn the policy refuses is answered with the refusal and never reaches the model. Empty means every turn is answered."`
	Harness            *Harness                 `json:"harness,omitempty"`
	Instructions       *string                  `json:"instructions,omitempty"`
	Keyterms           *[]string                `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong, such as product or company names. Up to 100 terms, and providers that cannot be told about vocabulary ignore them."`
	KnowledgeNamespace *string                  `json:"knowledge_namespace,omitempty" doc:"What the agent may look things up in. Empty means it knows only what it was told."`
	Llm                *string                  `json:"llm,omitempty" doc:"The model holding the conversation."`
	Mode               *AgentMode               `json:"mode,omitempty"`
	Name               string                   `json:"name" doc:"What the config is called, which is unique among the customer's own."`
	ProgressiveTools   *bool                    `json:"progressive_tools,omitempty" doc:"Whether the agent is offered its plugin, MCP server and connector tools by the first line of each one's description, with its arguments' descriptions left out, and the first call to a tool returns its full description and input schema instead of running it. It saves context on an agent with many tools, at the cost of one more model turn for each tool a conversation uses. Off by default. Left out on an update, the stored setting stays."`
	AgentPlugins       *[]PluginEntry           `json:"agent_plugins,omitempty" doc:"Hosted MCP servers this agent may reach with the app's own login, named from the built-in catalog: an id alone, or an object naming it with how it is reached, such as linear's read-only endpoint and the scopes its login asks for."`
	PluginEvents       *[]PluginEvent           `json:"plugin_events,omitempty" maxItems:"32" doc:"MCP events the agent subscribes to on the plugins it names, with every login it holds to each. Each event that arrives opens a text conversation of its own, as whoever's login it came through."`
	McpServers         *[]McpServer             `json:"mcp_servers,omitempty" maxItems:"16" doc:"MCP servers outside the plugin catalog, opened by their URL with no login. Their tools are offered as <name>__<tool>."`
	Connectors         *[]AgentConnectorBinding `json:"connectors,omitempty" maxItems:"64" doc:"The connectors whose tools this agent may call, each under an alias unique within the config and different from every plugin and MCP server it names. Omitted or null on an update, the bindings stored stay as they are, so a client that does not know this field cannot clear it by saving; an empty list removes them all. A binding to a connector the app cannot see, or a fixed binding to a connection that is not the app's own or is to another connector, is refused."`
	Channels           *AgentChannels           `json:"channels,omitempty" doc:"Lines this agent answers on besides Stream Chat: a WhatsApp number, a number to text, an iMessage line. Each must be connected with POST /v1/agents/channels."`
	Sandbox            *Sandbox                 `json:"sandbox,omitempty"`
	SandboxOptions     *SandboxOptions          `json:"sandbox_options,omitempty"`
	Search             *string                  `json:"search,omitempty" doc:"What the agent finds out today's answers with, as a provider/model or a capability shortcut. Empty leaves the default, and a deployment that routes no search offers the tool to nobody either way."`
	Skills             *[]string                `json:"skills,omitempty" doc:"Skill names, either the customer's own or one of the built-in think, recall and explain. Omit for the built-in set."`
	Speed              *float64                 `json:"speed,omitempty" doc:"Rate of delivery, 1 being the voice's own. Zero or absent leaves it there. A config that names one is only routed to voices that can be sped up, and one outside that voice's own range is refused." minimum:"0" example:"0.9"`
	Sts                *string                  `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Stt                *string                  `json:"stt,omitempty" doc:"A provider/model or a capability shortcut. Empty leaves the default, and a text agent ignores it."`
	Tags               *map[string]string       `json:"tags,omitempty" doc:"Cost labels, carried onto every request a session using it makes."`
	ThinkingLlm        *string                  `json:"thinking_llm,omitempty" doc:"The slower model a voice agent hands its skills to, while the voice model keeps talking. Only a voice agent names one: a text agent runs everything, skills included, on its llm. Empty leaves the default thinking model."`
	Tts                *string                  `json:"tts,omitempty"`
	UserPlugins        *[]PluginEntry           `json:"user_plugins,omitempty" doc:"Hosted MCP servers each end user connects with their own account, named from the built-in catalog like agent_plugins. The agent asks for the login in the conversation, as a plugin_authorization attachment, the first time it needs one."`
	Video              *SessionVideo            `json:"video,omitempty"`
	VisibleTools       *[]string                `json:"visible_tools,omitempty" doc:"Tools whose steps end users see on a persistent conversation's replies, as tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its arguments or result. A shown tool whose result is exactly {\"status\":\"answered\",\"citations\":[{\"id\",\"title\",\"url\",\"citation\"}]} also adds those citations to the reply's sources. Empty shows search and web_search." maxItems:"64"`
	Voice              *string                  `json:"voice,omitempty" doc:"Provider-specific voice id."`
}

// AgentConfig is the AgentConfig schema.
type AgentConfig struct {
	CreatedAt          time.Time                `json:"created_at"`
	Dispatch           *AgentDispatch           `json:"dispatch,omitempty"`
	EpisodeCards       *bool                    `json:"episode_cards,omitempty" doc:"Whether each phone call under this agent writes an episode card into the caller's omni-channel, and each session on a thread channel or a phone call starts with the person's other cards."`
	Greeting           *string                  `json:"greeting,omitempty"`
	Guardrail          *string                  `json:"guardrail,omitempty"`
	Harness            *Harness                 `json:"harness,omitempty"`
	Id                 string                   `json:"id"`
	Instructions       *string                  `json:"instructions,omitempty"`
	Keyterms           *[]string                `json:"keyterms,omitempty"`
	KnowledgeNamespace *string                  `json:"knowledge_namespace,omitempty"`
	Llm                *string                  `json:"llm,omitempty"`
	Mode               AgentMode                `json:"mode"`
	Name               string                   `json:"name"`
	ProgressiveTools   *bool                    `json:"progressive_tools,omitempty" doc:"Whether tools from plugins, MCP servers and connectors are offered by a summary, the first call to each returning its full description instead of running it."`
	AgentPlugins       *[]PluginEntry           `json:"agent_plugins,omitempty"`
	PluginEvents       *[]PluginEvent           `json:"plugin_events,omitempty"`
	McpServers         *[]McpServer             `json:"mcp_servers,omitempty"`
	Connectors         *[]AgentConnectorBinding `json:"connectors,omitempty" doc:"The bindings exactly as they were written. Absent when there are none."`
	Channels           *AgentChannels           `json:"channels,omitempty"`
	Sandbox            *Sandbox                 `json:"sandbox,omitempty"`
	SandboxOptions     *SandboxOptions          `json:"sandbox_options,omitempty"`
	Search             *string                  `json:"search,omitempty"`
	Skills             *[]string                `json:"skills,omitempty"`
	Speed              *float64                 `json:"speed,omitempty"`
	Sts                *string                  `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Stt                *string                  `json:"stt,omitempty"`
	SyncHash           *string                  `json:"sync_hash,omitempty" doc:"Fingerprint of the last directory synced onto this config. Empty if it was never synced from a directory."`
	Tags               *map[string]string       `json:"tags,omitempty"`
	ThinkingLlm        *string                  `json:"thinking_llm,omitempty"`
	Tts                *string                  `json:"tts,omitempty"`
	UpdatedAt          time.Time                `json:"updated_at"`
	UserPlugins        *[]PluginEntry           `json:"user_plugins,omitempty"`
	Video              *SessionVideo            `json:"video,omitempty"`
	VisibleTools       *[]string                `json:"visible_tools,omitempty"`
	Voice              *string                  `json:"voice,omitempty"`
}

// AgentConnectorBinding is a connector whose tools an agent config may call, under an alias.
//
// The alias pattern is the prototype's connectorAliasPattern (internal/api/connectors.go:54 on
// codex/connector-support at cf62af0d), which gives no reason for the length of 63; that is
// unverified. A trailing _ is refused because the offered name would not split back: alias a_
// and tool search are offered as a___search, and a cut at the first aliasSeparator reads alias
// a and tool _search. A config holds at most 64 bindings (the maxItems on each connectors
// field), the cap visible_tools already has, and 128 tools is the most OpenAI documents taking
// in one request. Neither was measured for connectors, so both are unverified. The 30000 ms ceiling is the prototype's
// (internal/api/connectors.go:1176 at cf62af0d), and nothing there says why 30 seconds; it is
// unverified.
type AgentConnectorBinding struct {
	Name        string                   `json:"name" pattern:"^[a-z]([a-z0-9_-]{0,61}[a-z0-9-])?$" doc:"The alias, unique within the config: a lowercase letter, then up to 62 lowercase letters, digits, - or _, never __ and not ending in _. The model is offered each tool as <name>__<tool>, split back at the first __, so a __ inside the alias or a _ at its end would split it in the wrong place."`
	ConnectorId string                   `json:"connector_id" doc:"A connector definition the app can see: a built-in, or one of its own, whose id starts with custom_."`
	Connection  AgentConnectorSelection  `json:"connection"`
	Tools       []ConnectorToolGrant     `json:"tools" maxItems:"128" nullable:"false" doc:"The exact tools allowed, each named once. There is no wildcard, and an empty list grants none. A session binding may grant a tool by name alone, which pins its schema per connection on first use."`
	Required    *bool                    `json:"required,omitempty" default:"false" doc:"Whether a session needs this connector. A required one that cannot be opened fails the session; an optional one is left out of it."`
	TimeoutMs   *int                     `json:"timeout_ms,omitempty" minimum:"1" maximum:"30000" doc:"How long one tool call may take, in milliseconds. Omitted, the session's default applies."`
	Events      *[]ConnectorBindingEvent `json:"events,omitempty" maxItems:"32" doc:"MCP events the binding's fixed connection is subscribed to, each opening a text conversation from the config when it arrives. Subscribed when the connection is next validated. Only a fixed binding may declare events: a session binding's connection is picked when a session opens, and an event arrives with no session open."`
	Policy      *ConnectorBindingPolicy  `json:"policy,omitempty" doc:"How the binding's tool calls behave around speech and interruptions. Omitted, a call is cancelled at the provider when the turn is interrupted, and the agent picks its own words while it runs."`
}

// ConnectorBindingPolicy is a binding's policy envelope, read back as it was written.
type ConnectorBindingPolicy struct {
	PreSpeech   *string               `json:"pre_speech,omitempty" minLength:"1" doc:"What the agent says while one of the binding's tools runs, such as \"Let me pull up your calendar.\", in place of the phrase it picks itself when the model reached for the tool without a word. A voice session with a separate voice says it; every session reports it on tool_started."`
	OnInterrupt *ConnectorOnInterrupt `json:"on_interrupt,omitempty" doc:"What an interruption of the turn does to a call in flight. Omitted, the call goes on and the caller is told its result when it comes, unless they withdraw what they asked for, which stops it."`
	Cancellable *bool                 `json:"cancellable,omitempty" doc:"Whether the provider is told to stop a call the session stopped waiting for. Omitted is true. False leaves it running once the session stops waiting, for a tool that is not safe to stop halfway, such as a payment; the binding's timeout still ends it and tells the provider to stop it. It does not matter with on_interrupt wait, whose call the session always waits for."`
}

func (*ConnectorBindingPolicy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How a binding's tool calls behave around speech and interruptions. Every field is " +
		"optional, and a field left out keeps today's behaviour."
	return schema
}

// ConnectorOnInterrupt is what an interruption does to a binding's call in flight.
type ConnectorOnInterrupt string

const (
	ConnectorOnInterruptCancel ConnectorOnInterrupt = store.InterruptCancel
	ConnectorOnInterruptWait   ConnectorOnInterrupt = store.InterruptWait
)

func (ConnectorOnInterrupt) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorOnInterrupt", "cancel stops waiting for the call when the turn "+
		"is interrupted, and tells the provider to stop it unless cancellable is false. wait lets the call "+
		"finish, up to the binding's timeout, and its result goes into the conversation for the next turn.",
		string(ConnectorOnInterruptCancel), string(ConnectorOnInterruptWait))
}

// ConnectorBindingEvent is one MCP event a binding subscribes to on its connection's server.
// The cap of 32 on a binding's events is plugin_events', a choice that is unverified.
type ConnectorBindingEvent struct {
	Event        string          `json:"event" minLength:"1" doc:"The event's name, as the server's events/list gives it, such as issue.created."`
	Arguments    *map[string]any `json:"arguments,omitempty" doc:"The event's filters, as its inputSchema describes them."`
	Instructions *string         `json:"instructions,omitempty" doc:"What the agent does with the event when it arrives, added to its instructions for that conversation."`
}

func (*ConnectorBindingEvent) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One MCP event a binding subscribes to on its fixed connection. Each one that " +
		"arrives opens a text conversation from the config, as the app, with the event's data as the " +
		"first thing said to it."
	return schema
}

func (*AgentConnectorBinding) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connector whose tools an agent config may call, under an alias. The binding is " +
		"the grant: only the tools it lists are offered, each pinned to the schema it was reviewed against, or " +
		"for a tool a session binding grants by name alone, to the schema its connection first offered it with."
	return schema
}

// AgentConnectorSelection is which connection a binding's tools are called through.
type AgentConnectorSelection struct {
	Type         AgentConnectorSelectionType `json:"type"`
	ConnectionId *string                     `json:"connection_id,omitempty" doc:"Required for fixed, and refused for session."`
}

func (*AgentConnectorSelection) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which connection a binding's tools are called through."
	return schema
}

// AgentConnectorSelectionType is whether a binding's connection is fixed or picked per session.
type AgentConnectorSelectionType string

const (
	AgentConnectorSelectionTypeFixed   AgentConnectorSelectionType = "fixed"
	AgentConnectorSelectionTypeSession AgentConnectorSelectionType = "session"
)

func (AgentConnectorSelectionType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AgentConnectorSelectionType", "fixed is the app's own connection named by "+
		"connection_id, the same for every session. session is the connection the session's verified end "+
		"user picks when the session is created, which has to be their own.",
		string(AgentConnectorSelectionTypeFixed), string(AgentConnectorSelectionTypeSession))
}

// ConnectorToolGrant is one tool a binding allows. The digest is a SHA-256, 32 bytes in
// lowercase hex, which is what the prototype took of a tool's name, description and input
// schema (ToolSchemaDigest, internal/mcp/mcp.go:172-184 at cf62af0d) and checked a grant's
// digest against (connectorToolDigestPattern, internal/api/connectors.go:56).
//
// A session binding may leave the digest out (Kanat's decision of 2026-10-08, AI-816): its
// connection is each person's own, and a provider may write the person into a tool's
// description, as Slack does with the signed-in user's id, so no one digest fits everyone.
// The session that first opens a connection pins the digest it lists then
// (session.Manager.pinGrants, store.ConnectorToolPin).
type ConnectorToolGrant struct {
	Name         string `json:"name" minLength:"1" doc:"The tool as the connector names it."`
	SchemaDigest string `json:"schema_digest,omitempty" pattern:"^[a-f0-9]{64}$" doc:"The SHA-256 of the tool's name, description and input schema, as 64 lowercase hex characters. A tool whose schema has changed since no longer matches and is not offered. Required on a fixed binding. A session binding may leave it out: the first session that opens a person's connection pins the digest the provider lists then, later sessions offer the tool only while it still matches, and a reconnect pins again."`
}

func (*ConnectorToolGrant) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One tool a binding allows."
	return schema
}

func (*AgentConfigRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["visible_tools"].Items.MaxLength = itemLimit(128)
	return schema
}

type listSkillsRequest struct {
	ConfigId optionalParam[string] `query:"config_id" doc:"Only the skills belonging to this agent config. Omit for every skill the customer has, across all of their agents."`
}

type listSkillsResponse struct {
	Body []Skill `nullable:"false"`
}

type createSkillRequest struct {
	Body *SkillRequest `required:"true"`
}

type createSkillResponse struct {
	Body Skill
}

type getSkillRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}

type getSkillResponse struct {
	Body Skill
}

type updateSkillRequest struct {
	Id   string        `path:"id" doc:"The resource, as returned when it was created."`
	Body *SkillRequest `required:"true"`
}

type updateSkillResponse struct {
	Body Skill
}

type deleteSkillRequest struct {
	Id string `path:"id" doc:"The resource, as returned when it was created."`
}
