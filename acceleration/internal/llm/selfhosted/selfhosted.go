// Package selfhosted reaches open-weight models we deploy on Baseten ourselves, one
// deployment per model, each with its own endpoint. The recipes are in deploy/: lfm-2.5,
// qwen-3.6, nemotron-bolt, phonellm-mini and phonellm-max.
//
// Each deployment is its own provider because each is its own endpoint and its own bill,
// and routing has to be able to tell them apart from the same weights on a shared host.
package selfhosted

import (
	"errors"
	"os"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const apiKeyEnvVar = "BASETEN_API_KEY"

// flowPurpose is the flow controller's requests, which are a classification and answered
// at zero.
const flowPurpose = "flow"

// quickPurposes are the requests with a deadline too short to think in: a flow ruling and
// a line composed for a pause, which is dropped after three seconds.
var quickPurposes = map[string]bool{flowPurpose: true, "compose": true}

// Deployment is one model on its own Baseten endpoint.
type Deployment struct {
	Provider      string
	BaseURLEnvVar string
	ModelID       string
	// Temperature is what a conversation request that names none is answered at: the
	// card's recommendation where it gives one for agents, else Gemma's 0.2. A vLLM
	// deployment otherwise answers at the checkpoint's generation_config, up to 1.0.
	Temperature float64
	// Fields are sent on every request, for a model whose template needs them whatever
	// the request says.
	Fields map[string]any
	// Prefill opens the replies to quickPurposes, for a model whose template has no
	// switch for thinking. The conversation's own replies still think.
	Prefill string
}

// Deployments are keyed by provider name.
var Deployments = map[string]Deployment{
	"liquid": {
		Provider: "liquid", BaseURLEnvVar: "LFM_BASE_URL", ModelID: "qvm0plxq",
		Temperature: 0.1,
		// LFM2.5 thinks before every reply, which costs a quick request its deadline. A
		// closed thinking block skips it, and enable_thinking tells the server's reasoning
		// parser that what follows is the answer.
		Prefill: "<think>\n</think>\n",
		// Left to itself it thinks for a second or two before a reply; a sentence or two
		// keeps what thinking buys at a fraction of the wait.
		Fields: map[string]any{"thinking_token_budget": 48},
	},
	"qwen-baseten": {
		Provider: "qwen-baseten", BaseURLEnvVar: "QWEN_BASE_URL", ModelID: "qjjnx4lq",
		Temperature: 0.2,
	},
	"nemotron-bolt": {
		Provider: "nemotron-bolt", BaseURLEnvVar: "BOLT_BASE_URL", ModelID: "wdlnomk3",
		Temperature: 0.2,
	},
	"phonellm-mini": {
		Provider: "phonellm-mini", BaseURLEnvVar: "PHONELLM_MINI_BASE_URL", ModelID: "qz49pvpw",
		Temperature: 0,
	},
	// Max is trained to write one sentence of reasoning before each reply, and its
	// reasoning parser reads the mode from the request, so both switches go on every one.
	"phonellm-max": {
		Provider: "phonellm-max", BaseURLEnvVar: "PHONELLM_MAX_BASE_URL", ModelID: "qrp0dk93",
		Temperature: 0,
		Fields: map[string]any{"chat_template_kwargs": map[string]any{
			"enable_thinking":           true,
			"truncate_history_thinking": false,
		}},
	},
}

// Builders are the constructors the LLM registry registers, one per deployment.
func Builders() map[string]func(openweights.Options) (*openaicompat.LLM, error) {
	builders := make(map[string]func(openweights.Options) (*openaicompat.LLM, error), len(Deployments))
	for name, deployment := range Deployments {
		builders[name] = deployment.New
	}
	return builders
}

// New builds the provider for one deployment.
func (d Deployment) New(options openweights.Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = keypool.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, stack.Wrap(errors.New(d.Provider + ": " + apiKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		return nil, stack.Wrap(errors.New(d.Provider + ": model is required"))
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(d.BaseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = "https://model-" + d.ModelID + ".api.baseten.co/environments/production/sync/v1"
	}
	thinking := openweights.RequestFields(options.Model, options.Thinking)
	capabilities := openweights.Capabilities(options.Model, options.Thinking, options.ReasoningEffort)
	if d.Fields != nil {
		capabilities.StreamsReasoning = true
	}
	return openaicompat.New(openaicompat.Options{
		Provider:     d.Provider,
		Model:        options.Model,
		APIKey:       options.APIKey,
		BaseURL:      options.BaseURL,
		Capabilities: capabilities,
		Prefill: func(params llm.ResponseParams) string {
			if !quickPurposes[params.Purpose] {
				return ""
			}
			return d.Prefill
		},
		RequestFields: func(params llm.ResponseParams, effort string) map[string]any {
			fields := map[string]any{}
			if thinking != nil {
				for key, value := range thinking(params, effort) {
					fields[key] = value
				}
			}
			for key, value := range d.Fields {
				fields[key] = value
			}
			if d.Prefill != "" && quickPurposes[params.Purpose] {
				fields["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
			}
			if params.Temperature == nil {
				fields["temperature"] = d.Temperature
				if params.Purpose == flowPurpose {
					fields["temperature"] = 0.0
				}
			}
			return fields
		},
		Logger: options.Logger,
	})
}
