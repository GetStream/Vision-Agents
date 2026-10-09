package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// SyncAgentRequest is an agent directory as it is on disk.
type SyncAgentRequest struct {
	Name          string                     `json:"name" doc:"What the config is called, which is also the directory's name."`
	Hash          string                     `json:"hash" doc:"A fingerprint of the directory. A second sync with the same hash does nothing."`
	CheckChanges  *bool                      `json:"check_changes,omitempty" doc:"Refuse the sync, with unsynced_changes, when somebody has changed one of the settings it would write since the last sync -- in the dashboard, say. A client that asks for this shows the person what changed (GET /v1/agents/configs/{id}/changes) and syncs again with base_change once they have decided. Omitted, the sync writes over whatever is there, which is what a process syncing on startup wants."`
	BaseChange    *string                    `json:"base_change,omitempty" doc:"The newest change the caller has already seen, as last_change named it. Everything up to it is taken as decided, so the sync is not refused for it again."`
	Instructions  *string                    `json:"instructions,omitempty"`
	Guardrail     *string                    `json:"guardrail,omitempty" doc:"The directory's guardrail.md, whole: frontmatter saying how to screen a turn, then the policy in prose. Empty means every turn is answered."`
	Skills        *[]SkillRequest            `json:"skills,omitempty"`
	Knowledge     *[]KnowledgeDocument       `json:"knowledge,omitempty"`
	KnowledgeUrls *[]KnowledgeUrlDeclaration `json:"knowledge_urls,omitempty" doc:"The pages the directory's knowledge/urls.yaml declares. They are subscribed to in the same knowledge base as the files, so one lookup covers both."`
	Simulations   *[]SimulationDeclaration   `json:"simulations,omitempty" doc:"The simulations the directory's simulations/*.yaml declare. Sent, they are the whole of the agent's simulations: each is found by name, and one no longer declared is deleted. Left out, the stored ones are left alone."`
	Mode          *AgentMode                 `json:"mode,omitempty"`
	Stt           *string                    `json:"stt,omitempty"`
	Tts           *string                    `json:"tts,omitempty"`
	Sts           *string                    `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Voice         *string                    `json:"voice,omitempty"`
	Llm           *string                    `json:"llm,omitempty"`
	Video         *SessionVideo              `json:"video,omitempty"`
	Subagent      *string                    `json:"subagent,omitempty" doc:"Only a voice agent names one: a text agent runs everything on its llm."`
	Search        *string                    `json:"search,omitempty"`
	Greeting      *Greeting                  `json:"greeting,omitempty"`
	Plugins       *[]PluginEntry             `json:"plugins,omitempty" doc:"Plugins the agent reaches: a catalog id, or an object naming it with how it is reached. The app connects each once, unless its entry sets user: then each end user connects it with their own account, from the conversation, the first time the agent needs it."`
	PluginEvents  *[]PluginEvent             `json:"plugin_events,omitempty" maxItems:"32" doc:"MCP events the agent subscribes to on its plugins, each opening a text conversation when it arrives."`
	McpServers    *[]McpServer               `json:"mcp_servers,omitempty" maxItems:"16" doc:"MCP servers outside the plugin catalog, opened by their URL with no login."`
	Channels      *AgentChannels             `json:"channels,omitempty" doc:"Lines this agent answers on besides Stream Chat, each a number the app connected."`
	Connectors    *[]AgentConnectorBinding   `json:"connectors,omitempty" maxItems:"64" doc:"The connectors agent.yaml binds. Sent, they are the whole of the agent's bindings and replace the ones stored, an empty list removing them all. Left out, the stored ones are left alone."`
	Keyterms      *[]string                  `json:"keyterms,omitempty"`
	Sandbox       *Sandbox                   `json:"sandbox,omitempty"`
	Harness       *Harness                   `json:"harness,omitempty"`
	Dispatch      *AgentDispatch             `json:"dispatch,omitempty"`
	// SandboxOptions is how the sandbox is built. Left out keeps what is stored.
	SandboxOptions *SandboxOptions    `json:"sandbox_options,omitempty"`
	Tags           *map[string]string `json:"tags,omitempty"`
	Tools          *AgentTools        `json:"tools,omitempty" doc:"How plugin, MCP server and connector tools are offered. A setting left out keeps what is stored."`
}

func (*SyncAgentRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An agent directory as it is on disk. Everything after the simulations " +
		"is what the directory's declaration decides rather than what it holds, and a setting " +
		"left out leaves whatever is stored, so a model chosen in the dashboard survives a sync " +
		"that says nothing about it."
	return schema
}

// KnowledgeUrlDeclaration is a page an agent directory declares.
type KnowledgeUrlDeclaration struct {
	Url          string  `json:"url" example:"https://example.com/pricing"`
	Title        *string `json:"title,omitempty" example:"Pricing"`
	Description  *string `json:"description,omitempty"`
	RefreshHours *int    `json:"refresh_hours,omitempty" minimum:"1" example:"24" doc:"How often the page is read again on its own, in hours. Omit it and the page is read on every sync that changes the directory, never on a schedule."`
}

func (*KnowledgeUrlDeclaration) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A page an agent directory declares, in the knowledge base named after it."
	return schema
}

// SimulationDeclaration is one simulation an agent directory declares. It runs against the
// agent being synced, so it names no config.
type SimulationDeclaration struct {
	Name         string             `json:"name" minLength:"1" doc:"Unique among the agent's simulations, and what a sync finds it again by."`
	Scenario     string             `json:"scenario" minLength:"1" doc:"What the caller wants, in your own words and over as many turns as it takes."`
	Assertion    string             `json:"assertion" minLength:"1" doc:"What has to be true at the end for a run to have passed."`
	Mode         *string            `json:"mode,omitempty" enum:"text,audio" doc:"Text when left out."`
	Variations   *int               `json:"variations,omitempty" minimum:"1" maximum:"10" doc:"How many ways of asking the same thing one run tries."`
	MaxTurns     *int               `json:"max_turns,omitempty" minimum:"1" maximum:"200" doc:"How many times the caller may speak. Twelve when left out."`
	CallerTarget *string            `json:"caller_target,omitempty"`
	JudgeTarget  *string            `json:"judge_target,omitempty"`
	CallerStt    *string            `json:"caller_stt,omitempty"`
	CallerTts    *string            `json:"caller_tts,omitempty"`
	CallerVoice  *string            `json:"caller_voice,omitempty"`
	Tags         *map[string]string `json:"tags,omitempty"`
}

func (*SimulationDeclaration) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A simulation an agent directory declares in simulations/*.yaml. It " +
		"runs against the agent being synced."
	return schema
}

// SyncAgentResult is the config a sync left stored.
type SyncAgentResult struct {
	Unchanged bool        `json:"unchanged" doc:"True when the hash matched and nothing was written."`
	Config    AgentConfig `json:"config"`
	Warnings  []string    `json:"warnings,omitempty" doc:"What was stored but will not work yet, such as a channel line the app has not connected."`
}

type syncAgentRequest struct {
	Body SyncAgentRequest
}

type syncAgentResponse struct {
	Body SyncAgentResult
}

func (s *Server) registerSync(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "syncAgent",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sync",
		Summary:     "Store an agent directory's instructions, skills, knowledge, simulations and settings",
		Description: "Reads as \"this is what the agent is\", from a directory of agent.yaml, " +
			"instructions.md, skills/, knowledge/ and simulations/. The hash is a fingerprint of " +
			"that directory: a second call with the same hash does nothing, so a process that " +
			"syncs on startup is cheap when nothing has changed.\n\n" +
			"agent.yaml decides the models, the voice and the rest of a config, so an agent kept " +
			"in a repository needs nothing written by hand. A setting it leaves out is left alone " +
			"rather than blanked.\n\n" +
			"knowledge/ is the whole of the knowledge base named after the agent, and simulations/ " +
			"the whole of its simulations: a file taken out of the directory is taken out of the " +
			"backend on the next sync.\n\n" +
			"A directory is not the only thing that writes an agent: somebody may have changed " +
			"one of the same settings in the dashboard since the last sync. Send " +
			"`check_changes` and such a sync is refused with `unsynced_changes` instead of " +
			"writing over them -- only when it really would write over them, so a directory " +
			"that already holds what the dashboard says syncs without complaint. Read the " +
			"changes from `GET /v1/agents/configs/{id}/changes`, let the person decide, and " +
			"sync again with `base_change` to go ahead.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as stored, or as it already was"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict},
	}, s.syncAgent)
}

// syncAgent stores an agent directory: its instructions, skills, knowledge and simulations,
// and the settings its declaration decided.
//
// The hash is a fingerprint of that directory. A second call with the same hash does
// nothing, so a process that syncs on startup is cheap when nothing has changed.
func (s *Server) syncAgent(ctx context.Context, request *syncAgentRequest) (*syncAgentResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}

	body := request.Body
	name := strings.TrimSpace(body.Name)
	hash := strings.TrimSpace(body.Hash)
	if name == "" {
		return nil, invalidRequest("an agent config needs a name")
	}
	if hash == "" {
		return nil, invalidRequest("a hash is required, so a second sync can do nothing")
	}

	if s.store == nil {
		return nil, errNoConfigs
	}
	if message, ok := syncComplaint(body); !ok {
		return nil, invalidRequest(message)
	}

	existing, found, err := s.configs.AgentConfigByName(ctx, customerID, name)
	if err != nil {
		return nil, err
	}
	if found && existing.SyncHash == hash {
		drifted, err := s.driftedSinceSync(ctx, customerID, existing.ID, body)
		if err != nil {
			return nil, err
		}
		if !drifted {
			return &syncAgentResponse{Body: SyncAgentResult{Unchanged: true, Config: agentConfigOf(existing)}}, nil
		}
	}
	// Before anything is written, for the same reason as the simulations in syncComplaint.
	if message, ok, err := s.unboundConnectors(ctx, customerID, body.Connectors); err != nil {
		return nil, err
	} else if !ok {
		return nil, invalidRequest(message)
	}
	config := existing
	if !found {
		config = store.AgentConfig{CustomerID: customerID, Name: name}
	}
	applySettings(&config, body)
	if message, ok := textSubagentComplaint(&config, body.Subagent); !ok {
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
	warnings, message, ok := s.channelsWarnings(ctx, config)
	if !ok {
		return nil, invalidRequest(message)
	}
	unclientable, err := s.pluginClientWarnings(ctx, config)
	if err != nil {
		return nil, err
	}
	warnings = append(warnings, unclientable...)
	if message, ok := pluginAliasComplaint(config); !ok {
		return nil, invalidRequest(message)
	}

	skills := skillsOf(body.Skills)
	named := make([]string, 0, len(skills))
	for _, skill := range skills {
		named = append(named, strings.TrimSpace(skill.Name))
	}

	// Before anything is written, and before the knowledge is filled: a caller that asked
	// to be told is told instead of having its answer half applied. What the sync would
	// store is worked out first, because the question is what it would write over rather
	// than whether anything was edited.
	if found && value(body.CheckChanges) {
		prospective := config
		prospective.Instructions = value(body.Instructions)
		prospective.Guardrail = value(body.Guardrail)
		prospective.Skills = named
		refusal, conflicting, err := s.syncConflict(ctx, customerID, existing, prospective, body)
		if err != nil {
			return nil, err
		}
		if conflicting {
			return nil, refusal
		}
	}

	documents := documentsOf(body.Knowledge)
	namespace := ""
	if len(documents) > 0 {
		if s.knowledge == nil {
			return nil, errNoKnowledge
		}
		namespace = name
		if _, _, err := s.fillKnowledge(ctx, customerID, namespace, documents, nil); err != nil {
			return nil, invalidRequest(err.Error())
		}
	}
	if body.KnowledgeUrls != nil && len(*body.KnowledgeUrls) > 0 {
		if s.pages == nil {
			return nil, errNoKnowledgeURLs
		}
		namespace = name
		for _, page := range *body.KnowledgeUrls {
			wanted := urls.Subscription{Namespace: namespace, URL: page.Url}
			wanted.Title = value(page.Title)
			wanted.Description = value(page.Description)
			wanted.RefreshHours = value(page.RefreshHours)
			if _, err := s.pages.Add(ctx, customerID, wanted); err != nil {
				return nil, invalidRequest(err.Error())
			}
		}
	}
	// The directory is the whole of what its knowledge base holds, so a file taken out of
	// it is taken out of the base too, including the last one.
	if s.knowledge != nil && (namespace != "" || existing.KnowledgeNamespace == name) {
		if err := s.forgetKnowledge(ctx, customerID, name, documents); err != nil {
			return nil, err
		}
	}

	config.Instructions = value(body.Instructions)
	config.Guardrail = value(body.Guardrail)
	config.Skills = named
	config.KnowledgeNamespace = namespace
	config.SyncHash = hash

	if found {
		if err := s.configs.UpdateAgentConfig(ctx, &config); err != nil {
			return nil, storeFailure(err, errAgentNameTaken)
		}
	} else {
		if err := s.configs.CreateAgentConfig(ctx, &config); err != nil {
			return nil, storeFailure(err, errAgentNameTaken)
		}
	}
	var was any
	if found {
		was = agentConfigOf(existing)
	}
	synced := auditDiff(was, agentConfigOf(config))

	// The skills and simulations belong to the config, so they are written after it: a new
	// agent has no id to hang them off until it has been stored.
	if len(skills) > 0 {
		if err := s.upsertSkills(ctx, customerID, config.ID, skills); err != nil {
			return nil, invalidRequest(err.Error())
		}
	}
	if body.Simulations != nil {
		if err := s.replaceSimulations(ctx, customerID, config.ID, *body.Simulations); err != nil {
			return nil, err
		}
	}
	// Recorded last, after the skills the directory brought with it, because what it marks
	// is the moment the whole directory and the whole stored agent agreed. An entry written
	// before the skills would leave each of them looking like an edit made since the sync,
	// and the next sync would refuse itself over its own writes. It is recorded whether or
	// not anything moved: the point is the moment, not the diff.
	s.audit(ctx, auditRecord{
		ResourceType: store.AuditAgentConfig, ResourceID: config.ID, ResourceName: config.Name,
		Action: store.AuditSynced, Changes: synced,
	})
	s.pluginEvents.Changed(customerID, config.ID)
	return &syncAgentResponse{Body: SyncAgentResult{Unchanged: false, Config: agentConfigOf(config), Warnings: warnings}}, nil
}

// syncComplaint reports what is wrong with the settings a directory declared, if
// anything. It is the same reading configComplaint does, since a directory decides the
// same things a config written by hand does.
func syncComplaint(body SyncAgentRequest) (string, bool) {
	if _, ok := modeOf(body.Mode); !ok {
		return fmt.Sprintf("an agent is either %s or %s", store.AgentModeVoice, store.AgentModeText), false
	}
	if complaint, ok := guardrailComplaint(body.Guardrail); !ok {
		return complaint, false
	}
	if len(keytermsOf(body.Keyterms)) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}
	if _, ok := sandboxOf(body.Sandbox); !ok {
		return fmt.Sprintf("there is no sandbox provider called %q", *body.Sandbox), false
	}
	if _, ok := harnessOf(body.Harness); !ok {
		return fmt.Sprintf("there is no harness called %q", *body.Harness), false
	}
	if complaint, ok := dispatchComplaint(body.Dispatch); !ok {
		return complaint, false
	}
	if complaint, ok := connectorBindingsComplaint(body.Connectors); !ok {
		return complaint, false
	}
	if complaint, ok := sandboxOptionsComplaint(body.SandboxOptions); !ok {
		return complaint, false
	}
	// Simulations are checked here, before anything is written, since a config stored under
	// the new hash would make the next sync skip the simulations that failed.
	named := map[string]bool{}
	for _, simulation := range simulationsOf(body.Simulations) {
		name := strings.TrimSpace(simulation.Name)
		if named[name] {
			return fmt.Sprintf("two simulations are called %q", name), false
		}
		named[name] = true
		if simulation.Tags != nil {
			if err := routing.Tags(*simulation.Tags).Validate(); err != nil {
				return fmt.Sprintf("simulation %q: %s", name, err), false
			}
		}
	}
	return "", true
}

// replaceSimulations makes the config's simulations exactly the ones declared, finding each
// by name so its runs stay attached to it.
func (s *Server) replaceSimulations(ctx context.Context, customerID, configID string, declared []SimulationDeclaration) error {
	all, err := s.store.CustomerSimulations(ctx, customerID)
	if err != nil {
		return err
	}
	stored := map[string]store.Simulation{}
	for _, simulation := range all {
		if simulation.ConfigID == configID {
			stored[simulation.Name] = simulation
		}
	}

	for _, declaration := range declared {
		simulation := storedSimulation(SimulationRequest{
			Name:         declaration.Name,
			ConfigId:     configID,
			Scenario:     declaration.Scenario,
			Assertion:    declaration.Assertion,
			Mode:         (*SimulationRequestMode)(declaration.Mode),
			Variations:   declaration.Variations,
			MaxTurns:     declaration.MaxTurns,
			CallerTarget: declaration.CallerTarget,
			JudgeTarget:  declaration.JudgeTarget,
			CallerStt:    declaration.CallerStt,
			CallerTts:    declaration.CallerTts,
			CallerVoice:  declaration.CallerVoice,
			Tags:         declaration.Tags,
		}, customerID)
		existing, ok := stored[simulation.Name]
		delete(stored, simulation.Name)
		if ok {
			simulation.ID = existing.ID
			if err := s.store.UpdateSimulation(ctx, &simulation); err != nil {
				return err
			}
			continue
		}
		if err := s.store.CreateSimulation(ctx, &simulation); err != nil {
			return err
		}
	}

	for _, simulation := range stored {
		if err := s.store.DeleteSimulation(ctx, customerID, simulation.ID); err != nil {
			return err
		}
	}
	return nil
}

func simulationsOf(list *[]SimulationDeclaration) []SimulationDeclaration {
	if list == nil {
		return nil
	}
	return *list
}

// applySettings writes onto a config what the directory's declaration decided. Only what
// was sent is applied: a directory that says nothing about a model leaves the one already
// stored, so a target chosen in the dashboard survives a sync.
func applySettings(config *store.AgentConfig, body SyncAgentRequest) {
	if body.Video != nil {
		config.VideoSource = override(config.VideoSource, body.Video.Source)
		config.VideoMaxFrames = override(config.VideoMaxFrames, body.Video.MaxFrames)
	}
	if body.Mode != nil && *body.Mode != "" {
		mode, _ := modeOf(body.Mode)
		config.Mode = mode
	}
	if body.Stt != nil {
		config.STT = *body.Stt
	}
	if body.Tts != nil {
		config.TTS = *body.Tts
	}
	if body.Sts != nil {
		config.STS = *body.Sts
	}
	if body.Voice != nil {
		config.Voice = *body.Voice
	}
	if body.Llm != nil {
		config.LLM = *body.Llm
	}
	if body.Subagent != nil {
		config.Subagent = *body.Subagent
	}
	if body.Search != nil {
		config.Search = *body.Search
	}
	if body.Greeting != nil {
		config.Greeting, config.GreetingMode = greetingOf(body.Greeting)
	}
	if body.Plugins != nil {
		config.Plugins = pluginEntriesOf(*body.Plugins)
	}
	if body.Connectors != nil {
		config.Connectors = storedBindings(*body.Connectors)
	}
	if body.PluginEvents != nil {
		config.PluginEvents = pluginEventsOf(body.PluginEvents)
	}
	if body.McpServers != nil {
		config.MCPServers = mcpServersOf(body.McpServers)
	}
	config.ProgressiveTools = progressiveOf(config.ProgressiveTools, body.Tools)
	if body.Channels != nil {
		config.Channels = channelsOf(body.Channels)
	}
	if body.Keyterms != nil {
		config.Keyterms = keytermsOf(body.Keyterms)
	}
	if body.Sandbox != nil {
		box, _ := sandboxOf(body.Sandbox)
		config.Sandbox = box
	}
	if body.Harness != nil {
		config.Harness, _ = harnessOf(body.Harness)
	}
	applyDispatch(config, body.Dispatch)
	if body.SandboxOptions != nil {
		config.SandboxOptions = sandboxConfigOf(body.SandboxOptions)
	}
	if body.Tags != nil {
		config.Tags = *body.Tags
	}
}

func skillsOf(list *[]SkillRequest) []SkillRequest {
	if list == nil {
		return nil
	}
	return *list
}

func documentsOf(list *[]KnowledgeDocument) []KnowledgeDocument {
	if list == nil {
		return nil
	}
	return *list
}

func (s *Server) upsertSkills(ctx context.Context, customerID, configID string, skills []SkillRequest) error {
	names := make([]string, 0, len(skills))
	for _, skill := range skills {
		// The directory names the skill and the config owns it, so the caller does not
		// repeat the config id per skill and it is filled in here.
		skill.ConfigId = configID
		if message, ok := skillComplaint(skill); !ok {
			return stack.Wrap(errors.New(message))
		}
		names = append(names, strings.TrimSpace(skill.Name))
	}

	stored, err := s.configs.SkillsNamed(ctx, customerID, configID, names)
	if err != nil {
		return err
	}
	known := map[string]store.Skill{}
	for _, skill := range stored {
		known[skill.Name] = skill
	}

	for _, skill := range skills {
		skill.ConfigId = configID
		row := storedSkill(skill, customerID)
		if existing, ok := known[row.Name]; ok {
			row.ID = existing.ID
			row.CreatedAt = existing.CreatedAt
			if err := s.configs.UpdateSkill(ctx, &row); err != nil {
				return err
			}
			s.audit(ctx, auditRecord{
				ResourceType: store.AuditSkill, ResourceID: row.ID, ResourceName: row.Name,
				AgentID: configID, Action: store.AuditUpdated,
				Changes: auditDiff(skillOf(existing), skillOf(row)),
			})
			continue
		}
		if err := s.configs.CreateSkill(ctx, &row); err != nil {
			return err
		}
		s.audit(ctx, auditRecord{
			ResourceType: store.AuditSkill, ResourceID: row.ID, ResourceName: row.Name,
			AgentID: configID, Action: store.AuditCreated, Changes: auditDiff(nil, skillOf(row)),
		})
	}
	return nil
}

// SimulationRequestMode is the SimulationRequestMode schema.
type SimulationRequestMode string

// Defines values for SimulationRequestMode.
const (
	SimulationRequestModeAudio SimulationRequestMode = "audio"
	SimulationRequestModeText  SimulationRequestMode = "text"
)

// Valid indicates whether the value is a known member of the SimulationRequestMode enum.
func (e SimulationRequestMode) Valid() bool {
	switch e {
	case SimulationRequestModeAudio:
		return true
	case SimulationRequestModeText:
		return true
	default:
		return false
	}
}
