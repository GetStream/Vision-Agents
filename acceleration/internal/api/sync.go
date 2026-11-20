package api

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agentdir"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// SyncAgentRequest is an agent directory as it is on disk.
type SyncAgentRequest struct {
	Name              string                     `json:"name" doc:"What the config is called, which is also the directory's name."`
	Hash              string                     `json:"hash" doc:"A fingerprint of the directory. A second sync with the same hash does nothing."`
	CheckChanges      *bool                      `json:"check_changes,omitempty" doc:"Refuse the sync, with unsynced_changes, when somebody has changed one of the settings it would write since the last sync -- in the dashboard, say. A client that asks for this shows the person what changed (GET /v1/agents/configs/{id}/changes) and syncs again with base_change once they have decided. Omitted, the sync writes over whatever is there, which is what a process syncing on startup wants."`
	BaseChange        *string                    `json:"base_change,omitempty" doc:"The newest change the caller has already seen, as last_change named it. Everything up to it is taken as decided, so the sync is not refused for it again."`
	Instructions      *string                    `json:"instructions,omitempty"`
	Guardrail         *string                    `json:"guardrail,omitempty" doc:"The directory's guardrail.md, whole: frontmatter saying how to screen a turn, then the policy in prose. Empty means every turn is answered."`
	UiInstructions    *string                    `json:"ui_instructions,omitempty" doc:"What teaches the agent's model to answer with UI, which the SDK compiles from agent.yaml's ui key. Kept apart from instructions, so a session opened with ui false is not told them. Left out, the agent is taught none."`
	Skills            *[]SkillRequest            `json:"skills,omitempty"`
	Knowledge         *[]KnowledgeDocument       `json:"knowledge,omitempty"`
	KnowledgeUrls     *[]KnowledgeUrlDeclaration `json:"knowledge_urls,omitempty" doc:"The pages the directory's knowledge/urls.yaml declares. They are subscribed to in the same knowledge base as the files, so one lookup covers both."`
	Simulations       *[]SimulationDeclaration   `json:"simulations,omitempty" doc:"The simulations the directory's simulations/*.yaml declare. Sent, they are the whole of the agent's simulations: each is found by name, and one no longer declared is deleted. Left out, the stored ones are left alone."`
	Role              *AgentRole                 `json:"role,omitempty"`
	Mode              *AgentMode                 `json:"mode,omitempty"`
	Region            *string                    `json:"region,omitempty" pattern:"^[a-z][a-z0-9-]{0,31}$" patternDescription:"a letter and up to 31 of a-z, 0-9 and hyphens, such as us or eu" example:"eu" doc:"agent.yaml's region: where the agent's work is done, us or eu for now. Every model its sessions run is served there and nowhere else, its knowledge base is kept there, and its sandbox is made there. A sync is refused (400), before anything is written, when a model the agent names has no instance there, naming each and the region, when its knowledge base is already kept in another region, since a knowledge base is never moved, or when it names a sandbox and none runs there. Left out, a new agent is in us and a stored one keeps its region."`
	Stt               *string                    `json:"stt,omitempty"`
	Tts               *string                    `json:"tts,omitempty"`
	Sts               *string                    `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Voice             *string                    `json:"voice,omitempty" doc:"The voice the agent speaks in, as TtsOptions.voice takes it: a provider's own voice id, or one of your voices, with the custom: prefix to mean only yours. A native agent (sts) takes the vendor's own voice name instead."`
	Llm               *string                    `json:"llm,omitempty"`
	Video             *SessionVideo              `json:"video,omitempty"`
	Subagent          *string                    `json:"subagent,omitempty" doc:"Only a voice agent names one: a text agent runs everything on its llm."`
	Search            *string                    `json:"search,omitempty"`
	Image             *string                    `json:"image,omitempty" doc:"agent.yaml's image: what the agent draws pictures with. Left out, the stored one stays; an empty string removes it."`
	Greeting          *Greeting                  `json:"greeting,omitempty"`
	Plugins           *[]retiredPlugin           `json:"plugins,omitempty" deprecated:"true" doc:"No longer read: plugins were replaced by connectors, so name the provider under connectors. A non-empty list is refused (400); an empty one names nothing."`
	PluginEvents      *[]map[string]any          `json:"plugin_events,omitempty" deprecated:"true" doc:"No longer read: plugins were replaced by connectors, so name the provider under connectors. A non-empty list is refused (400); an empty one names nothing."`
	McpServers        *[]map[string]any          `json:"mcp_servers,omitempty" deprecated:"true" doc:"No longer read: a server named by URL is now a custom connector, so create one for its URL and name it under connectors. A non-empty list is refused (400); an empty one names nothing."`
	Channels          *AgentChannels             `json:"channels,omitempty" doc:"Lines this agent answers on besides Stream Chat, each a number the app connected."`
	ChatChannels      *[]AgentChatChannel        `json:"chat_channels,omitempty" maxItems:"16" doc:"agent.yaml's chat_channels: the Stream Chat channel types of the app's own Stream app this agent answers in besides agent channels, as AgentConfigRequest's chat_channels names them, with the same rules. Sent, they replace the stored list whole, an empty list removing it. Left out, the stored list stays."`
	Connectors        *[]AgentConnectorBinding   `json:"connectors,omitempty" maxItems:"64" doc:"The connectors agent.yaml binds. Sent, they are the whole of the agent's bindings and replace the ones stored, an empty list removing them all. Left out, the stored ones are left alone."`
	LeftOutConnectors *[]LeftOutConnector        `json:"left_out_connectors,omitempty" maxItems:"64" doc:"The connectors agent.yaml names that the client left out of connectors, because the app has no connection by the id or label they name. Stored as sent and served on the config, so the dashboard can say what the agent is missing. A non-empty list is refused without connectors: a sync that sends connectors replaces the stored ones with these, or with none when this is left out, and a sync that sends neither leaves them alone. At most 64, as connectors."`
	Keyterms          *[]string                  `json:"keyterms,omitempty"`
	Sandbox           *Sandbox                   `json:"sandbox,omitempty"`
	Harness           *Harness                   `json:"harness,omitempty"`
	Dispatch          *AgentDispatch             `json:"dispatch,omitempty"`
	// SandboxOptions is how the sandbox is built. Left out keeps what is stored.
	SandboxOptions *SandboxOptions    `json:"sandbox_options,omitempty"`
	Tags           *map[string]string `json:"tags,omitempty"`
	Agents         *[]ChildAgent      `json:"agents,omitempty" maxItems:"16" doc:"agent.yaml's agents: the app's other text agents this one may hand work to, each as a tool, by name. Sent, they replace the stored list whole, an empty list removing it; left out, the stored list stays. An agent named must already exist, so a directory is synced after the agents it names."`
	Tools          *[]ToolGroup       `json:"tools,omitempty" maxItems:"64" doc:"agent.yaml's tools: the functions an SDK provides that the agent may offer, in named groups, as AgentConfigRequest's tools names them, with the same rules. Sent, they replace the stored groups whole, an empty list removing them. Left out, the stored ones stay."`
	Calls          *AgentCalls        `json:"calls,omitempty" doc:"The tools an agent on a phone call is offered: end the call, press keys, transfer. A tool left out keeps what is stored; a tool sent replaces its settings whole."`
	Scheduling     *AgentScheduling   `json:"scheduling,omitempty" doc:"Whether the agent may schedule tasks for its users, and the defaults those tasks take. A setting left out keeps what is stored."`
	ResumeHistory  *ResumeHistory     `json:"resume_history,omitempty" doc:"How a conversation is handed back to the model when a session carries it on. A setting left out keeps what is stored."`
	FloorControl   *FloorControl      `json:"floor_control,omitempty" doc:"agent.yaml's floor_control: the model that controls the agent's floor, and how long a caller may wait once they stop. Sent, it replaces the stored choice whole, and an empty object is no choice again; left out or null, the stored one stays. An agent that chose none gets the router's default: the flow controller, or stream/chorus-asr at the server's default point where the router defaults every agent to it; {model: flow-controller} keeps the flow controller whatever that default. stream/chorus-asr on a router without a chorus-asr server is refused (400), and so is a max_wait_ms shorter than every point the chorus-asr server publishes, before anything is written. When the server cannot be asked the wait is kept and a warning says so, as it does of a point that fails the server's dictation guard."`
	EndOfTurn      any                `json:"end_of_turn,omitempty" hidden:"true"` // refused: floor_control's max_wait_ms replaced it
}

func (*SyncAgentRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An agent directory as it is on disk. Everything after the simulations " +
		"is what the directory's declaration decides rather than what it holds, and a setting " +
		"left out leaves whatever is stored, so a model chosen in the dashboard survives a sync " +
		"that says nothing about it."
	return schema
}

// LeftOutConnector is a connector agent.yaml names that a client left out of a sync,
// because the app has no connection by the id or label it names.
//
// The caps are the ones the same values have elsewhere: 63 is the longest alias
// AgentConnectorBinding's name pattern allows, 64 the longest custom connector id
// (custom_ and 57 more, internal/api/connectors.go) and 120 the longest connection label
// (CreateConnectionRequest). A built-in connector id is shorter than a custom one.
type LeftOutConnector struct {
	Name        string `json:"name" minLength:"1" maxLength:"63" doc:"The alias agent.yaml gave the connector."`
	ConnectorId string `json:"connector_id" minLength:"1" maxLength:"64" doc:"The connector agent.yaml named, or the alias when it named none."`
	Connection  string `json:"connection" minLength:"1" maxLength:"120" doc:"The connection id or label agent.yaml named, which no connection of the app has."`
}

func (*LeftOutConnector) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connector agent.yaml names that a sync left out, because the app " +
		"has no connection by the id or label it names."
	return schema
}

// leftOutOf turns the left-out connectors a sync sent into what a config stores.
func leftOutOf(sent *[]LeftOutConnector) []store.LeftOutConnector {
	if sent == nil || len(*sent) == 0 {
		return nil
	}
	stored := make([]store.LeftOutConnector, 0, len(*sent))
	for _, connector := range *sent {
		stored = append(stored, store.LeftOutConnector{
			Name: connector.Name, ConnectorID: connector.ConnectorId, Connection: connector.Connection,
		})
	}
	return stored
}

// KnowledgeUrlDeclaration is a page an agent directory declares.
type KnowledgeUrlDeclaration struct {
	Url          string  `json:"url" maxLength:"2048" example:"https://example.com/pricing"`
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
	Unchanged bool         `json:"unchanged" doc:"True when the hash matched and nothing was written."`
	Config    AgentConfig  `json:"config"`
	Warnings  []string     `json:"warnings,omitempty" doc:"What was stored but will not work yet, such as a channel line the app has not connected."`
	Changes   []SyncChange `json:"changes,omitempty" nullable:"false" doc:"What the sync did, or with dry_run would do, to what the agent holds beside its config: each skill, knowledge document, knowledge url and simulation it created, updated or deleted. One it leaves as it is is not listed, and nothing is for a sync that found the hash unchanged."`
}

// SyncChange is one thing a sync writes beside the config.
type SyncChange struct {
	ResourceType string `json:"resource_type" enum:"skill,knowledge,knowledge_url,simulation" doc:"What it is, named as GET /v1/agents/configs/{id}/changes names a skill, a knowledge document and a knowledge url, and simulation, which that record does not keep."`
	Name         string `json:"name" doc:"Which one: a skill's name, a knowledge document's source, a knowledge url's url, a simulation's name."`
	Action       string `json:"action" enum:"created,updated,deleted"`
}

func (*SyncChange) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One skill, knowledge document, knowledge url or simulation a sync creates, updates or deletes."
	return schema
}

// syncChanges works out what a sync would write beside the config, before it writes any of
// it: the same question each write below answers for itself, asked together so a dry run can
// say it.
func (s *Server) syncChanges(
	ctx context.Context, customerID string, existing store.AgentConfig, body SyncAgentRequest, fill knowledgeFill, forgets bool,
) ([]SyncChange, error) {
	var changes []SyncChange
	say := func(kind, name, action string) {
		changes = append(changes, SyncChange{ResourceType: kind, Name: name, Action: action})
	}
	written := func(held bool) string {
		if held {
			return store.AuditUpdated
		}
		return store.AuditCreated
	}

	var skills []store.Skill
	var simulations []store.Simulation
	if existing.ID != "" {
		var err error
		if skills, err = s.store.CustomerSkills(ctx, customerID, existing.ID); err != nil {
			return nil, err
		}
		all, err := s.store.CustomerSimulations(ctx, customerID)
		if err != nil {
			return nil, err
		}
		for _, simulation := range all {
			if simulation.ConfigID == existing.ID {
				simulations = append(simulations, simulation)
			}
		}
	}
	for _, skill := range skillsOf(body.Skills) {
		was, held := skillNamedStored(skills, strings.TrimSpace(skill.Name))
		if !held || !sameSkill(was, storedSkill(skill, was.CustomerID)) {
			say(store.AuditSkill, strings.TrimSpace(skill.Name), written(held))
		}
	}

	for _, document := range fill.written {
		_, held := fill.before[document.Source]
		say(store.AuditKnowledge, document.Source, written(held))
	}
	if forgets {
		stored, err := s.store.CustomerKnowledgeDocuments(ctx, customerID, body.Name)
		if err != nil {
			return nil, err
		}
		for _, document := range stored {
			kept := slices.ContainsFunc(documentsOf(body.Knowledge), func(sent KnowledgeDocument) bool {
				return strings.TrimSpace(sent.Source) == document.Source && strings.TrimSpace(sent.Text) != ""
			})
			if !kept {
				say(store.AuditKnowledge, document.Source, store.AuditDeleted)
			}
		}
	}
	if pages := value(body.KnowledgeUrls); len(pages) > 0 {
		subscribed, err := s.pages.List(ctx, customerID, body.Name)
		if err != nil {
			return nil, err
		}
		for _, page := range pages {
			at := slices.IndexFunc(subscribed, func(was store.KnowledgeURL) bool { return was.URL == strings.TrimSpace(page.Url) })
			if at < 0 || subscribed[at].DeclaredTitle != value(page.Title) || subscribed[at].Description != value(page.Description) ||
				subscribed[at].RefreshHours != value(page.RefreshHours) {
				say(store.AuditKnowledgeURL, strings.TrimSpace(page.Url), written(at >= 0))
			}
		}
	}

	if body.Simulations != nil {
		declared := map[string]bool{}
		for _, declaration := range *body.Simulations {
			name := strings.TrimSpace(declaration.Name)
			declared[name] = true
			at := slices.IndexFunc(simulations, func(was store.Simulation) bool { return was.Name == name })
			if at < 0 || !sameSimulation(simulations[at], declaration) {
				say("simulation", name, written(at >= 0))
			}
		}
		for _, simulation := range simulations {
			if !declared[simulation.Name] {
				say("simulation", simulation.Name, store.AuditDeleted)
			}
		}
	}
	return changes, nil
}

// sameSimulation reports whether a declaration says what a stored simulation is, with what
// the store fills in for what a declaration leaves out.
func sameSimulation(was store.Simulation, declared SimulationDeclaration) bool {
	return was.Scenario == strings.TrimSpace(declared.Scenario) && was.Assertion == strings.TrimSpace(declared.Assertion) &&
		was.Mode == cmp.Or(value(declared.Mode), store.SimulationText) &&
		was.Variations == max(value(declared.Variations), 1) &&
		was.MaxTurns == cmp.Or(value(declared.MaxTurns), store.DefaultSimulationTurns) &&
		was.CallerTarget == value(declared.CallerTarget) && was.JudgeTarget == value(declared.JudgeTarget) &&
		was.CallerSTT == value(declared.CallerStt) && was.CallerTTS == value(declared.CallerTts) &&
		was.CallerVoice == value(declared.CallerVoice) && maps.Equal(was.Tags, value(declared.Tags))
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
	return s.sync(ctx, directorySync{body: request.Body})
}

// sync stores what a sync says the agent is: the request a client sent, or the one a
// directory's files amount to.
func (s *Server) sync(ctx context.Context, read directorySync) (*syncAgentResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConfigs
	}
	body := read.body
	name := strings.TrimSpace(body.Name)
	hash := strings.TrimSpace(body.Hash)
	if name == "" {
		return nil, invalidRequest("an agent config needs a name")
	}
	if hash == "" {
		return nil, invalidRequest("a hash is required, so a second sync can do nothing")
	}

	// A sync of files is told everything its settings may not be at once, each where
	// agent.yaml says it. One sent as fields is refused at the first, as it always was.
	var problems agentdir.Problems
	refuse := func(setting, message string) error {
		if read.directory == nil {
			return invalidRequest(message)
		}
		line, column := read.directory.Locate(setting)
		problems = append(problems, agentdir.Problem{File: agentdir.AgentFile, Line: line, Column: column, Message: message})
		return nil
	}
	for _, complaint := range syncComplaints(ctx, body) {
		if err := refuse(complaint.setting, complaint.message); err != nil {
			return nil, err
		}
	}
	// Before the hash is read, so a directory that still names any is refused rather than
	// found unchanged.
	if err := retiredComplaint(body.Plugins, body.PluginEvents, body.McpServers); err != nil {
		return nil, err
	}

	existing, found, err := s.configs.AgentConfigByName(ctx, customerID, name)
	if err != nil {
		return nil, err
	}
	// A router older than ui_instructions kept a sync's hash and dropped them, so the same sync
	// sent again with UI the config lacks is written rather than found unchanged.
	missingUI := value(body.UiInstructions) != "" && existing.UIInstructions == ""
	if found && existing.SyncHash == hash && !missingUI {
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
		if err := refuse("connectors", message); err != nil {
			return nil, err
		}
	}
	config := existing
	var stored *savedSandbox
	if found {
		stored = sandboxOfConfig(existing)
	} else {
		config = store.AgentConfig{CustomerID: customerID, Name: name}
	}
	applySettings(&config, body)
	// Named whatever the directory says, so the knowledge the sync writes is never written
	// into a base kept in another region than the agent's.
	config.Region = cmp.Or(config.Region, pricing.DefaultRegion)
	if message, ok := textSubagentComplaint(&config, body.Subagent); !ok {
		if err := refuse("subagent", message); err != nil {
			return nil, err
		}
	}
	warnings, message, ok := s.channelsWarnings(ctx, config)
	if !ok {
		if err := refuse("channels", message); err != nil {
			return nil, err
		}
	}
	warnings = append(read.warnings, warnings...)
	if message, ok := schedulingComplaint(config); !ok {
		if err := refuse("scheduling", message); err != nil {
			return nil, err
		}
	}
	if failure, refused := s.floorControlRefusal(body.FloorControl, body.EndOfTurn); refused {
		if read.directory == nil {
			return nil, failure
		}
		_ = refuse("floor_control", failure.Message)
	}
	if warning, message, ok := s.floorControlComplaint(ctx, config.FloorControl, existing.FloorControl); !ok {
		if err := refuse("floor_control", message); err != nil {
			return nil, err
		}
	} else if warning != "" && body.FloorControl != nil {
		warnings = append(warnings, warning)
	}
	if message, ok := callsComplaint(config.Calls); !ok {
		if err := refuse("calls", message); err != nil {
			return nil, err
		}
	}
	if message, ok := teamTagComplaint(config.Tags, existing.Tags); !ok {
		if err := refuse("tags", message); err != nil {
			return nil, err
		}
	}
	if message, ok := toolsComplaint(config); !ok {
		if err := refuse("tools", message); err != nil {
			return nil, err
		}
	}
	if message, ok, err := s.sandboxBindingsComplaint(ctx, customerID, config); err != nil {
		return nil, err
	} else if !ok {
		if err := refuse("connectors", message); err != nil {
			return nil, err
		}
	}
	if message, ok, err := s.sensitiveBindingsComplaint(ctx, customerID, config); err != nil {
		return nil, err
	} else if !ok {
		if err := refuse("connectors", message); err != nil {
			return nil, err
		}
	}
	if message, ok := s.liveSTTComplaint(config, existing.STT); !ok {
		if err := refuse("stt", message); err != nil {
			return nil, err
		}
	}
	var prior *store.AgentConfig
	if found {
		prior = &existing
	}
	if message, ok, err := s.childAgentsComplaint(ctx, customerID, &config, prior); err != nil {
		return nil, err
	} else if !ok {
		if err := refuse("agents", message); err != nil {
			return nil, err
		}
	}
	if failure, refused := s.sandboxLimitsComplaint(ctx, customerID, config, stored); refused {
		return nil, failure
	}
	// Asked of the knowledge base the sync is about to write, the one named after the agent.
	documents := documentsOf(body.Knowledge)
	knows := len(documents) > 0 || len(value(body.KnowledgeUrls)) > 0
	regional := config
	regional.KnowledgeNamespace = ""
	if knows {
		regional.KnowledgeNamespace = name
	}
	if message, ok, err := s.regionComplaint(ctx, regional, prior); err != nil {
		return nil, err
	} else if !ok {
		if err := refuse("region", message); err != nil {
			return nil, err
		}
	}

	skills := skillsOf(body.Skills)
	// After the settings, so a sandbox stored before and not repeated still counts.
	if message, ok := skillSandboxComplaint(skills, config.Sandbox != ""); !ok {
		if err := refuse("sandbox", message); err != nil {
			return nil, err
		}
	}
	if len(problems) > 0 {
		return nil, invalidDirectory(problems)
	}
	// Everything a write below would refuse is refused here, before the first of them: a sync
	// refused halfway is an agent half written, and a dry run answers as the sync would.
	named := make([]string, 0, len(skills))
	for _, skill := range skills {
		// The config owns its skills and may not be stored yet, so there is no id to name.
		skill.ConfigId = name
		if message, ok := skillComplaint(skill); !ok {
			return nil, invalidRequest(message)
		}
		if slices.Contains(named, strings.TrimSpace(skill.Name)) {
			return nil, invalidRequest(fmt.Sprintf("two skills are called %q", strings.TrimSpace(skill.Name)))
		}
		named = append(named, strings.TrimSpace(skill.Name))
	}
	pages := value(body.KnowledgeUrls)
	if len(documents) > 0 && s.knowledge == nil {
		return nil, errNoKnowledge
	}
	if len(pages) > 0 && s.pages == nil {
		return nil, errNoKnowledgeURLs
	}
	for _, document := range documents {
		if strings.TrimSpace(document.Source) == "" {
			return nil, invalidRequest("every document needs a source, which is what its passages are keyed by")
		}
	}
	for _, page := range pages {
		if err := urls.Check(urls.Subscription{URL: page.Url, RefreshHours: value(page.RefreshHours)}); err != nil {
			return nil, invalidRequest(err.Error())
		}
	}

	config.Instructions = value(body.Instructions)
	config.Guardrail = value(body.Guardrail)
	config.UIInstructions = value(body.UiInstructions)
	config.UISource = read.ui
	config.Skills = named
	// Before anything is written, and before the knowledge is filled: a caller that asked
	// to be told is told instead of having its answer half applied. What the sync would
	// store is worked out first, because the question is what it would write over rather
	// than whether anything was edited.
	if found && value(body.CheckChanges) {
		refusal, conflicting, err := s.syncConflict(ctx, customerID, existing, config, body)
		if err != nil {
			return nil, err
		}
		if conflicting {
			return nil, refusal
		}
	}
	namespace := ""
	if knows {
		namespace = name
	}
	config.KnowledgeNamespace = namespace
	config.SyncHash = hash
	// The knowledge is worked out, and the money for writing it asked for, before anything is
	// written, so a dry run is refused for it as the sync would be.
	var fill knowledgeFill
	if len(documents) > 0 {
		if fill, err = s.planKnowledge(ctx, customerID, config.Region, namespace, documents, nil); err != nil {
			return nil, err
		}
	}
	forgets := s.knowledge != nil && (namespace != "" || existing.KnowledgeNamespace == name)
	changes, err := s.syncChanges(ctx, customerID, existing, body, fill, forgets)
	if err != nil {
		return nil, err
	}
	if read.dryRun {
		// What a config that names no mode is stored as (store.normalizeConfig).
		config.Mode = cmp.Or(config.Mode, store.AgentModeVoice)
		return &syncAgentResponse{Body: SyncAgentResult{Config: agentConfigOf(config), Warnings: warnings, Changes: changes}}, nil
	}

	// The writes are the agent's work, so a base nothing was written to is made in its region
	// and one kept in another refuses them.
	if len(documents) > 0 {
		if err := s.writeKnowledge(ctx, customerID, config.Region, namespace, fill); err != nil {
			return nil, err
		}
	}
	if len(pages) > 0 {
		// A page is read later, by work naming no region, which writes where the base is kept.
		if s.knowledge != nil {
			if err := s.knowledgeFor(ctx, customerID, config.Region).Keep(ctx, namespace); err != nil {
				return nil, err
			}
		}
		for _, page := range pages {
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
	if forgets {
		if err := s.forgetKnowledge(ctx, customerID, name, documents); err != nil {
			return nil, err
		}
	}

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
			return nil, storeFailure(err, errSkillNameTaken)
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
	return &syncAgentResponse{Body: SyncAgentResult{Unchanged: false, Config: agentConfigOf(config), Warnings: warnings, Changes: changes}}, nil
}

// syncSetting is something wrong with one setting of a sync, by the name agent.yaml has for it.
type syncSetting struct {
	setting, message string
}

// syncComplaints reports everything wrong with the settings a directory declared, each
// with the agent.yaml setting it is about. It is the same reading configComplaint does,
// since a directory decides the same things a config written by hand does.
func syncComplaints(ctx context.Context, body SyncAgentRequest) (complaints []syncSetting) {
	say := func(setting, message string) { complaints = append(complaints, syncSetting{setting, message}) }
	if _, ok := modeOf(body.Mode); !ok {
		say("mode", fmt.Sprintf("an agent is either %s or %s", store.AgentModeVoice, store.AgentModeText))
	}
	if complaint, ok := agentRegionComplaint(body.Region); !ok {
		say("region", complaint)
	}
	if complaint, ok := guardrailComplaint(ctx, body.Guardrail); !ok {
		say("guardrail", complaint)
	}
	if len(keytermsOf(body.Keyterms)) > stt.MaxKeyterms {
		say("keyterms", fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms))
	}
	if _, ok := sandboxOf(body.Sandbox); !ok {
		say("sandbox", fmt.Sprintf("there is no sandbox provider called %q", *body.Sandbox))
	}
	if _, ok := harnessOf(body.Harness); !ok {
		say("harness", fmt.Sprintf("there is no harness called %q", *body.Harness))
	}
	if complaint, ok := dispatchComplaint(body.Dispatch); !ok {
		say("dispatch", complaint)
	}
	if body.LeftOutConnectors != nil && len(*body.LeftOutConnectors) > 0 && body.Connectors == nil {
		say("connectors", "left_out_connectors goes with connectors: send connectors too, an empty list if none is bound")
	}
	if complaint, ok := connectorBindingsComplaint(body.Connectors); !ok {
		say("connectors", complaint)
	}
	if complaint, ok := chatChannelsComplaint(body.ChatChannels); !ok {
		say("chat_channels", complaint)
	}
	if complaint, ok := sandboxOptionsComplaint(body.SandboxOptions); !ok {
		say("sandbox_options", complaint)
	}
	// Simulations are checked here, before anything is written, since a config stored under
	// the new hash would make the next sync skip the simulations that failed.
	named := map[string]bool{}
	for _, simulation := range simulationsOf(body.Simulations) {
		name := strings.TrimSpace(simulation.Name)
		if named[name] {
			say("simulations", fmt.Sprintf("two simulations are called %q", name))
		}
		named[name] = true
		if simulation.Tags != nil {
			if err := routing.Tags(*simulation.Tags).Validate(); err != nil {
				say("simulations", fmt.Sprintf("simulation %q: %s", name, err))
			}
		}
	}
	return complaints
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
	if body.Role != nil {
		config.Role = string(*body.Role)
	}
	if body.Mode != nil && *body.Mode != "" {
		mode, _ := modeOf(body.Mode)
		config.Mode = mode
	}
	if body.Region != nil {
		config.Region = *body.Region
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
	if body.Image != nil {
		config.Image = *body.Image
	}
	if body.Greeting != nil {
		config.Greeting, config.GreetingMode = greetingOf(body.Greeting)
	}
	if body.Connectors != nil {
		config.Connectors = storedBindings(*body.Connectors)
		config.LeftOutConnectors = leftOutOf(body.LeftOutConnectors)
	}
	if body.Tools != nil {
		config.ToolGroups = toolGroupsOf(*body.Tools)
	}
	config.Scheduling = schedulingOr(config.Scheduling, body.Scheduling)
	config.Calls = callsOr(config.Calls, body.Calls)
	config.ResumeHistory = resumeHistoryOr(config.ResumeHistory, body.ResumeHistory)
	config.FloorControl = floorControlOr(config.FloorControl, body.FloorControl)
	if body.Channels != nil {
		config.Channels = channelsOf(body.Channels)
	}
	if body.ChatChannels != nil {
		config.ChatChannels = chatChannelsOf(body.ChatChannels)
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
	if body.Agents != nil {
		config.Agents = childAgentsOf(body.Agents)
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
		// The config owns the skill, so the caller does not repeat its id per skill.
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
