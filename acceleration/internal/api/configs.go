package api

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/danielgtaylor/huma/v2"
)

// noConfigs is what the config and skill paths say on a deployment without a database.
// They are stored rather than computed, so there is nothing to serve without one.
const noConfigs = "agent configs are not available: no database configured"

type AgentConfig struct {
	Id                 string             `json:"id"`
	Name               string             `json:"name"`
	Mode               AgentMode          `json:"mode"`
	Stt                *string            `json:"stt,omitempty"`
	Tts                *string            `json:"tts,omitempty"`
	Sts                *string            `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Voice              *string            `json:"voice,omitempty"`
	Speed              *float64           `json:"speed,omitempty"`
	Llm                *string            `json:"llm,omitempty"`
	Video              *SessionVideo      `json:"video,omitempty"`
	Subagent           *string            `json:"subagent,omitempty"`
	Search             *string            `json:"search,omitempty"`
	Instructions       *string            `json:"instructions,omitempty"`
	Greeting           *string            `json:"greeting,omitempty"`
	Guardrail          *string            `json:"guardrail,omitempty"`
	Skills             *[]string          `json:"skills,omitempty"`
	Plugins            *[]string          `json:"plugins,omitempty"`
	Keyterms           *[]string          `json:"keyterms,omitempty"`
	VisibleTools       *[]string          `json:"visible_tools,omitempty"`
	KnowledgeNamespace *string            `json:"knowledge_namespace,omitempty"`
	Sandbox            *Sandbox           `json:"sandbox,omitempty"`
	Harness            *Harness           `json:"harness,omitempty"`
	Dispatch           *AgentDispatch     `json:"dispatch,omitempty"`
	Tags               *map[string]string `json:"tags,omitempty"`
	SyncHash           *string            `json:"sync_hash,omitempty" doc:"Fingerprint of the last directory synced onto this config. Empty if it was never synced from a directory."`
	CreatedAt          time.Time          `json:"created_at"`
	UpdatedAt          time.Time          `json:"updated_at"`
}

type AgentMode string

const (
	AgentModeText  AgentMode = "text"
	AgentModeVoice AgentMode = "voice"
)

// Valid indicates whether the value is a known member of the AgentMode enum.
func (e AgentMode) Valid() bool {
	switch e {
	case AgentModeText:
		return true
	case AgentModeVoice:
		return true
	default:
		return false
	}
}

type SessionVideo struct {
	Source    *string `json:"source,omitempty" doc:"Track or processor source. Omitted requires one unambiguous available source."`
	MaxFrames *int    `json:"max_frames,omitempty" doc:"Number of recent frames captured for a visual task. Default one." minimum:"1" maximum:"8"`
}

type Sandbox string

const (
	Daytona Sandbox = "daytona"
)

// Valid indicates whether the value is a known member of the Sandbox enum.
func (e Sandbox) Valid() bool {
	switch e {
	case Daytona:
		return true
	default:
		return false
	}
}

type Harness string

const (
	Default Harness = "default"
)

// Valid indicates whether the value is a known member of the Harness enum.
func (e Harness) Valid() bool {
	switch e {
	case Default:
		return true
	default:
		return false
	}
}

type AgentDispatch struct {
	IncomingCall *DispatchSetting `json:"incoming_call,omitempty" doc:"A call to one of the customer's numbers is handed to a dispatch worker. Every inbound call already is, since a number is not tied to an agent config."`
	Text         *DispatchSetting `json:"text,omitempty" doc:"An end user's message is handed to a dispatch worker, with the session it was written to, instead of being answered by the model. The worker answers by creating a response on that session with a server-side credential, passing the message's command_id when it has one; that is the only text the model answers."`
}

func (*AgentDispatch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the agent leaves to the customer's own server, which waits on /v1/dispatch. Omitted " +
		"settings are disabled."
	return schema
}

type DispatchSetting string

const (
	Disabled DispatchSetting = "disabled"
	Enabled  DispatchSetting = "enabled"
)

// Valid indicates whether the value is a known member of the DispatchSetting enum.
func (e DispatchSetting) Valid() bool {
	switch e {
	case Disabled:
		return true
	case Enabled:
		return true
	default:
		return false
	}
}

type AgentConfigRequest struct {
	Name               string             `json:"name" doc:"What the config is called, which is unique among the customer's own."`
	Mode               *AgentMode         `json:"mode,omitempty"`
	Stt                *string            `json:"stt,omitempty" doc:"A provider/model or a capability shortcut. Empty leaves the default, and a text agent ignores it."`
	Tts                *string            `json:"tts,omitempty"`
	Sts                *string            `json:"sts,omitempty" doc:"A speech-to-speech target: one native audio model that hears the caller and speaks back. Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade."`
	Voice              *string            `json:"voice,omitempty" doc:"Provider-specific voice id."`
	Speed              *float64           `json:"speed,omitempty" doc:"Rate of delivery, 1 being the voice's own. Zero or absent leaves it there. A config that names one is only routed to voices that can be sped up, and one outside that voice's own range is refused." minimum:"0" example:"0.9"`
	Llm                *string            `json:"llm,omitempty" doc:"The model holding the conversation."`
	Video              *SessionVideo      `json:"video,omitempty"`
	Subagent           *string            `json:"subagent,omitempty" doc:"The model that does the thinking. Empty means the voice model answers everything itself, and skills mean nothing."`
	Search             *string            `json:"search,omitempty" doc:"What the agent finds out today's answers with, as a provider/model or a capability shortcut. Empty leaves the default, and a deployment that routes no search offers the tool to nobody either way."`
	Instructions       *string            `json:"instructions,omitempty"`
	Greeting           *string            `json:"greeting,omitempty"`
	Guardrail          *string            `json:"guardrail,omitempty" doc:"A guardrail.md: frontmatter saying how a turn is screened - lcm, webhook or llm - then the policy in prose. A turn the policy refuses is answered with the refusal and never reaches the model. Empty means every turn is answered."`
	Skills             *[]string          `json:"skills,omitempty" doc:"Skill names, either the customer's own or one of the built-in think, recall and explain. Omit for the built-in set."`
	Plugins            *[]string          `json:"plugins,omitempty" doc:"Hosted MCP servers this agent may reach, named from the built-in catalog."`
	Keyterms           *[]string          `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong, such as product or company names. Up to 100 terms, and providers that cannot be told about vocabulary ignore them."`
	VisibleTools       *[]string          `json:"visible_tools,omitempty" doc:"Tools whose steps end users see on a persistent conversation's replies, as tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its arguments or result. A shown tool whose result is exactly {\"status\":\"answered\",\"citations\":[{\"id\",\"title\",\"url\",\"citation\"}]} also adds those citations to the reply's sources. Empty shows search and web_search." maxLength:"128" maxItems:"64"`
	KnowledgeNamespace *string            `json:"knowledge_namespace,omitempty" doc:"What the agent may look things up in. Empty means it knows only what it was told."`
	Sandbox            *Sandbox           `json:"sandbox,omitempty"`
	Harness            *Harness           `json:"harness,omitempty"`
	Dispatch           *AgentDispatch     `json:"dispatch,omitempty"`
	Tags               *map[string]string `json:"tags,omitempty" doc:"Cost labels, carried onto every request a session using it makes."`
}

type Skill struct {
	Id           string    `json:"id"`
	ConfigId     string    `json:"config_id"`
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	CaptureVideo *bool     `json:"capture_video,omitempty" doc:"Capture task-scoped visual evidence before reasoning."`
	Instructions string    `json:"instructions"`
	DeadlineMs   *int64    `json:"deadline_ms,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

type SkillRequest struct {
	ConfigId     string `json:"config_id" doc:"The agent config this skill belongs to. A skill is not shared: two agents that both need one have one each, so editing either leaves the other alone."`
	Name         string `json:"name" doc:"How the config names it, which is unique among that config's own skills."`
	Description  string `json:"description" doc:"The one line the fast model sees."`
	CaptureVideo *bool  `json:"capture_video,omitempty" doc:"Capture task-scoped visual evidence before reasoning."`
	Instructions string `json:"instructions" doc:"The full prompt, which only the subagent sees."`
	DeadlineMs   *int64 `json:"deadline_ms,omitempty" doc:"How long the work may run before it is abandoned. Zero is the default."`
}

type listAgentConfigsRequest struct {
	Name optionalParam[string] `query:"name" doc:"Narrow the list to the config with this name, which is how a name is resolved to a config. Names are unique per customer, so this answers with at most one."`
}

type agentConfigListResponse struct {
	Body []AgentConfig
}

type createAgentConfigRequest struct {
	Body AgentConfigRequest
}

type getAgentConfigRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type updateAgentConfigRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body AgentConfigRequest
}

type deleteAgentConfigRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type listSkillsRequest struct {
	ConfigID optionalParam[string] `query:"config_id" doc:"Only the skills belonging to this agent config. Omit for every skill the customer has, across all of their agents."`
}

type skillListResponse struct {
	Body []Skill
}

type createSkillRequest struct {
	Body SkillRequest
}

type skillResponse struct {
	Body Skill
}

type getSkillRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type updateSkillRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body SkillRequest
}

type deleteSkillRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

func (s *Server) registerConfigs(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listAgentConfigs",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs",
		Summary:     "The agent configs the calling customer holds",
		Description: "With a name this is how a name becomes a config, which is what lets a backend " +
			"say \"docs\" instead of an id it never chose.\n" +
			"Server-side only, as it always was: a config carries the instructions the agent " +
			"runs under, and those are not a page's business. A page addressing an agent by " +
			"name does not need this -- it sends the name on the create-session request and " +
			"the router resolves it, which is the same lookup without handing the " +
			"instructions over.",
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
		Description: "A config holds what a caller would otherwise repeat on every call: the models, " +
			"the voice, the instructions and which skills the subagent may be handed. What " +
			"is about one conversation rather than the agent behind it, the call id above " +
			"all, stays in the create-session request.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The config was stored"},
		},
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
		MaxBodyBytes: largeBody,
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
		Description: "Every field is written, so the body is what the config now is rather than what " +
			"changed about it. Sessions already running keep the configuration they started " +
			"with.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as it now is"},
		},
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
		MaxBodyBytes: largeBody,
	}, s.updateAgentConfig)
	huma.Register(api, huma.Operation{
		OperationID: "deleteAgentConfig",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/configs/{id}",
		Summary:     "Delete an agent config",
		Description: "Calls that already ran under it keep naming it, so the config stops being " +
			"usable rather than stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
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
		Description: "A skill belongs to one agent config. Two agents that both need the same kind of " +
			"work have one each, so editing what \"explain\" means for one leaves the other " +
			"alone. The built-in think, recall and explain need no row: a config may name " +
			"them without defining them.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The skill was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
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
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The skill as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.updateSkill)
	huma.Register(api, huma.Operation{
		OperationID: "deleteSkill",
		Method:      http.MethodDelete,
		Path:        "/v1/agents/skills/{id}",
		Summary:     "Delete a skill",
		Description: "Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"204": {Description: "The skill is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteSkill)
}

// listAgentConfigs returns the calling customer's configs, newest first.
func (s *Server) listAgentConfigs(ctx context.Context, request *listAgentConfigsRequest) (*agentConfigListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	// A name is resolved through the index rather than by reading every config and filtering
	// here, and an empty answer is an empty list rather than a 404: this is a list endpoint,
	// and a caller looking a name up is asking whether it is there.
	if named := request.Name.Value; named != "" {
		found, exists, err := s.store.AgentConfigByName(ctx, customerID, named)
		if err != nil {
			return nil, err
		}
		if !exists {
			return &agentConfigListResponse{Body: []AgentConfig{}}, nil
		}
		return &agentConfigListResponse{Body: []AgentConfig{agentConfigOf(found)}}, nil
	}

	stored, err := s.store.CustomerAgentConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]AgentConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, agentConfigOf(config))
	}
	return &agentConfigListResponse{Body: listed}, nil
}

// createAgentConfig stores a configuration sessions can be created from.
func (s *Server) createAgentConfig(ctx context.Context, request *createAgentConfigRequest) (*agentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if message, ok := configComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	config := storedConfig(request.Body, customerID)
	if err := s.store.CreateAgentConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &agentConfigResponse{Body: agentConfigOf(config)}, nil
}

// getAgentConfig returns one config.
func (s *Server) getAgentConfig(ctx context.Context, request *getAgentConfigRequest) (*agentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	config, err := s.store.AgentConfig(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}
	return &agentConfigResponse{Body: agentConfigOf(config)}, nil
}

// updateAgentConfig replaces a config with what it now is.
func (s *Server) updateAgentConfig(ctx context.Context, request *updateAgentConfigRequest) (*agentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if message, ok := configComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	existing, err := s.store.AgentConfig(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	config := storedConfig(request.Body, customerID)
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateAgentConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &agentConfigResponse{Body: agentConfigOf(config)}, nil
}

// deleteAgentConfig stops a config being usable.
func (s *Server) deleteAgentConfig(ctx context.Context, request *deleteAgentConfigRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	if err := s.store.DeleteAgentConfig(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}
	return nil, nil
}

// listSkills returns the calling customer's skills, newest first, or only the ones
// belonging to one agent config.
func (s *Server) listSkills(ctx context.Context, request *listSkillsRequest) (*skillListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	stored, err := s.store.CustomerSkills(ctx, customerID, request.ConfigID.Value)
	if err != nil {
		return nil, err
	}

	listed := make([]Skill, 0, len(stored))
	for _, skill := range stored {
		listed = append(listed, skillOf(skill))
	}
	return &skillListResponse{Body: listed}, nil
}

// createSkill defines a kind of work worth handing to the slower model.
func (s *Server) createSkill(ctx context.Context, request *createSkillRequest) (*skillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if message, ok := skillComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return nil, huma.Error400BadRequest(unknownConfig)
	}

	skill := storedSkill(request.Body, customerID)
	if err := s.store.CreateSkill(ctx, &skill); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &skillResponse{Body: skillOf(skill)}, nil
}

// getSkill returns one skill.
func (s *Server) getSkill(ctx context.Context, request *getSkillRequest) (*skillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	skill, err := s.store.Skill(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSkill)
	}
	return &skillResponse{Body: skillOf(skill)}, nil
}

// updateSkill replaces a skill with what it now is.
func (s *Server) updateSkill(ctx context.Context, request *updateSkillRequest) (*skillResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	if message, ok := skillComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}
	if _, err := s.store.AgentConfig(ctx, customerID, request.Body.ConfigId); err != nil {
		return nil, huma.Error400BadRequest(unknownConfig)
	}

	existing, err := s.store.Skill(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownSkill)
	}

	skill := storedSkill(request.Body, customerID)
	skill.ID = existing.ID
	skill.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateSkill(ctx, &skill); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &skillResponse{Body: skillOf(skill)}, nil
}

// deleteSkill stops a skill being usable.
func (s *Server) deleteSkill(ctx context.Context, request *deleteSkillRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}

	if err := s.store.DeleteSkill(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownSkill)
	}
	return nil, nil
}

// unknownConfig and unknownSkill are what a caller is told about a resource that is not
// theirs, which is the same thing they are told about one that never existed.
const (
	unknownConfig = "no such agent config"
	unknownSkill  = "no such skill"
)

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
	return dispatchComplaint(request.Dispatch)
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
		Subagent:           value(request.Subagent),
		Search:             value(request.Search),
		Instructions:       value(request.Instructions),
		Greeting:           value(request.Greeting),
		Guardrail:          value(request.Guardrail),
		KnowledgeNamespace: value(request.KnowledgeNamespace),
		Sandbox:            box,
		Harness:            named,
	}
	if request.Skills != nil {
		config.Skills = *request.Skills
	}
	if request.Plugins != nil {
		config.Plugins = *request.Plugins
	}
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
	rendered.Subagent = optional(config.Subagent)
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
	named := Harness(config.Harness)
	if named == "" {
		named = Default
	}
	rendered.Harness = &named
	rendered.Dispatch = dispatchOf(config)
	if len(config.Skills) > 0 {
		skills := config.Skills
		rendered.Skills = &skills
	}
	if len(config.Plugins) > 0 {
		named := config.Plugins
		rendered.Plugins = &named
	}
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
