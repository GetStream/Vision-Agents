package api

import (
	"context"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
)

// AgentConfigPatch is what changes about an agent config. Every field is optional, and one
// left out keeps what is stored.
type AgentConfigPatch struct {
	Name               *string            `json:"name,omitempty" minLength:"1" doc:"What the config is called, which is unique among the customer's own."`
	Mode               *AgentMode         `json:"mode,omitempty"`
	Stt                *string            `json:"stt,omitempty"`
	Tts                *string            `json:"tts,omitempty"`
	Sts                *string            `json:"sts,omitempty"`
	Voice              *string            `json:"voice,omitempty"`
	Speed              *float64           `json:"speed,omitempty" minimum:"0" doc:"The voice's rate of delivery, 1 being its own. Zero leaves it there."`
	Llm                *string            `json:"llm,omitempty"`
	Subagent           *string            `json:"subagent,omitempty"`
	Search             *string            `json:"search,omitempty"`
	Instructions       *string            `json:"instructions,omitempty"`
	Greeting           *string            `json:"greeting,omitempty"`
	Guardrail          *string            `json:"guardrail,omitempty" doc:"A guardrail.md: frontmatter saying how a turn is screened, then the policy in prose. An empty string removes the guardrail."`
	Skills             *[]string          `json:"skills,omitempty"`
	Plugins            *[]string          `json:"plugins,omitempty"`
	Keyterms           *[]string          `json:"keyterms,omitempty"`
	VisibleTools       *[]string          `json:"visible_tools,omitempty" maxItems:"64" doc:"Tools whose steps end users see on a persistent conversation's replies, as tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its arguments or result. A shown tool whose result is exactly {\"status\":\"answered\",\"citations\":[...]} also adds those citations to the reply's sources. An empty list shows search and web_search."`
	KnowledgeNamespace *string            `json:"knowledge_namespace,omitempty"`
	Sandbox            *Sandbox           `json:"sandbox,omitempty"`
	SandboxOptions     *SandboxOptions    `json:"sandbox_options,omitempty"`
	Harness            *Harness           `json:"harness,omitempty"`
	Dispatch           *AgentDispatch     `json:"dispatch,omitempty"`
	Tags               *map[string]string `json:"tags,omitempty"`
	Video              *SessionVideo      `json:"video,omitempty"`
}

func (*AgentConfigPatch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What changes about an agent config. A field left out keeps what is " +
		"stored, and an unknown one is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

func (AgentMode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AgentMode", "Whether the agent is spoken to or written to.",
		string(AgentModeVoice), string(AgentModeText))
}

func (Sandbox) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Sandbox", "Where the subagent may run code it writes.", string(Daytona))
}

func (Harness) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Harness", "Which harness the agent's sessions run. Set on the "+
		"agent, never on a session.", string(Default))
}

func (DispatchSetting) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "DispatchSetting", "Whether this kind of work is left to the "+
		"customer's own dispatch worker.", string(Enabled), string(Disabled))
}

type patchAgentConfigRequest struct {
	ID   string `path:"id" doc:"The config, as returned when it was created."`
	Body AgentConfigPatch
}

type agentConfigResponse struct {
	Body AgentConfig
}

func (s *Server) registerConfigPatch(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "patchAgentConfig",
		Method:      http.MethodPatch,
		Path:        "/v1/agents/configs/{id}",
		Summary:     "Change some of an agent config",
		Description: "Writes only the fields sent, so a guardrail can be set without restating " +
			"the instructions, skills and models beside it. Sessions already running keep the " +
			"configuration they started with.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The config as it now is"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound},
	}, s.patchAgentConfig)
}

// patchAgentConfig writes what a caller changed onto a stored config.
func (s *Server) patchAgentConfig(ctx context.Context, request *patchAgentConfigRequest) (*agentConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noConfigs)
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownConfig)
	}

	patch := request.Body
	if message, ok := configComplaint(AgentConfigRequest{
		Name:           override(config.Name, patch.Name),
		Mode:           patch.Mode,
		Keyterms:       patch.Keyterms,
		Sandbox:        patch.Sandbox,
		SandboxOptions: patch.SandboxOptions,
		Harness:        patch.Harness,
		Speed:          patch.Speed,
		Guardrail:      patch.Guardrail,
		VisibleTools:   patch.VisibleTools,
		Dispatch:       patch.Dispatch,
	}); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	config.Name = override(config.Name, patch.Name)
	if patch.Mode != nil {
		config.Mode, _ = modeOf(patch.Mode)
	}
	config.STT = override(config.STT, patch.Stt)
	config.TTS = override(config.TTS, patch.Tts)
	config.STS = override(config.STS, patch.Sts)
	config.Voice = override(config.Voice, patch.Voice)
	config.Speed = override(config.Speed, patch.Speed)
	config.LLM = override(config.LLM, patch.Llm)
	config.Subagent = override(config.Subagent, patch.Subagent)
	config.Search = override(config.Search, patch.Search)
	config.Instructions = override(config.Instructions, patch.Instructions)
	config.Greeting = override(config.Greeting, patch.Greeting)
	config.Guardrail = override(config.Guardrail, patch.Guardrail)
	config.Skills = override(config.Skills, patch.Skills)
	config.Plugins = override(config.Plugins, patch.Plugins)
	if patch.Keyterms != nil {
		config.Keyterms = keytermsOf(patch.Keyterms)
	}
	config.VisibleTools = override(config.VisibleTools, patch.VisibleTools)
	config.KnowledgeNamespace = override(config.KnowledgeNamespace, patch.KnowledgeNamespace)
	if patch.Sandbox != nil {
		config.Sandbox, _ = sandboxOf(patch.Sandbox)
	}
	if patch.SandboxOptions != nil {
		config.SandboxOptions = sandboxConfigOf(patch.SandboxOptions)
	}
	if patch.Harness != nil {
		config.Harness, _ = harnessOf(patch.Harness)
	}
	applyDispatch(&config, patch.Dispatch)
	config.Tags = override(config.Tags, patch.Tags)
	if patch.Video != nil {
		config.VideoSource = override(config.VideoSource, patch.Video.Source)
		config.VideoMaxFrames = override(config.VideoMaxFrames, patch.Video.MaxFrames)
	}
	// The config no longer matches the directory last synced onto it, so the next sync of
	// that directory writes it again rather than finding nothing changed.
	config.SyncHash = ""

	if err := s.configs.UpdateAgentConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &agentConfigResponse{Body: agentConfigOf(config)}, nil
}
