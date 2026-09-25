// Package openweights holds what the hosts of open-weight models have in common.
//
// The same weights are served by twenty different companies, so the model decides how it
// is asked to think rather than the host does: GLM reads a top-level thinking object,
// Qwen reads a chat template argument, Kimi reads a differently spelled one. Each host
// package names the model and this works out the rest, so the switch is written once.
package openweights

import (
	"errors"
	"log/slog"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
)

// Host is everything one host of open-weight models differs by. The weights are the
// same wherever they are served, so what is left is an endpoint and a key.
type Host struct {
	// Provider is the stable name used in routing config and stats.
	Provider string
	// APIKeyEnvVar holds the credentials when Options does not.
	APIKeyEnvVar string
	// BaseURLEnvVar overrides the endpoint, for a private deployment rather than the
	// public one.
	BaseURLEnvVar string
	// DefaultBaseURL is the public endpoint, up to and including /v1.
	DefaultBaseURL string
}

// Options configures one of this host's models.
type Options struct {
	APIKey string
	// Model is the id this host serves the weights under. Hosts disagree about how to
	// spell the same model, so the routing config names the one this host expects and
	// it travels unchanged.
	Model   string
	BaseURL string
	// Thinking lets the model reason before answering. It is off by default: reasoning
	// spends the whole token budget and most of the latency before the first word of
	// the answer, which is the wrong trade for a live conversation.
	Thinking bool
	// ReasoningEffort tunes how long the model thinks when Thinking is on, for the
	// families that offer a ladder. Empty leaves the model's own default.
	ReasoningEffort string
	Logger          *slog.Logger
}

// New builds a provider for one of this host's models, reading the API key from the
// environment when it is not given.
//
// There is no default model. A host here serves dozens of them under ids only it uses,
// so a model it guessed at would be a 404 rather than a sensible fallback.
func (h Host) New(options Options) (*openaicompat.LLM, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(h.APIKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New(h.Provider + ": " + h.APIKeyEnvVar + " is required")
	}
	if options.Model == "" {
		return nil, errors.New(h.Provider + ": model is required")
	}
	if options.BaseURL == "" {
		options.BaseURL = os.Getenv(h.BaseURLEnvVar)
	}
	if options.BaseURL == "" {
		options.BaseURL = h.DefaultBaseURL
	}

	return openaicompat.New(openaicompat.Options{
		Provider:      h.Provider,
		Model:         options.Model,
		APIKey:        options.APIKey,
		BaseURL:       options.BaseURL,
		Capabilities:  Capabilities(options.Model, options.Thinking, options.ReasoningEffort),
		RequestFields: RequestFields(options.Model, options.Thinking),
		Logger:        options.Logger,
	})
}

// Family is a set of open-weight models that share one thinking switch.
type Family string

const (
	// FamilyDeepSeek is DeepSeek's V4 line. Flash takes the switch as a chat template
	// argument; Pro takes it as a top-level object, which is why it is separate.
	FamilyDeepSeek    Family = "deepseek"
	FamilyDeepSeekPro Family = "deepseek-pro"
	FamilyGLM         Family = "glm"
	FamilyQwen        Family = "qwen"
	FamilyKimi        Family = "kimi"
	FamilyMiniMax     Family = "minimax"
	// FamilyNone is a model with no thinking to switch, and a model whose switch this
	// deployment has not seen answer. Both are asked for nothing rather than told
	// something the host may ignore.
	FamilyNone Family = ""
)

// FamilyOf reads the family out of a model id.
//
// Every host spells the same weights its own way -- zai-org/GLM-5.3-Flash, glm:5.3-flash,
// z-ai-glm-5-3-flash -- so this matches on the name inside the id rather than on a prefix
// or an exact id.
func FamilyOf(model string) Family {
	id := strings.ToLower(model)
	switch {
	case strings.Contains(id, "deepseek") && strings.Contains(id, "pro"):
		return FamilyDeepSeekPro
	case strings.Contains(id, "deepseek"):
		return FamilyDeepSeek
	case strings.Contains(id, "glm"):
		return FamilyGLM
	case strings.Contains(id, "qwen"):
		return FamilyQwen
	case strings.Contains(id, "kimi"):
		return FamilyKimi
	case strings.Contains(id, "minimax"):
		return FamilyMiniMax
	default:
		return FamilyNone
	}
}

// Efforts are the rungs a family answers to once it is thinking, and is empty for a
// family that offers no ladder.
func Efforts(model string) []string {
	switch FamilyOf(model) {
	case FamilyDeepSeekPro:
		return []string{"low", "medium", "high"}
	default:
		return nil
	}
}

// Thinks reports whether a family has a switch at all. A model that does not is left
// alone: Nemotron answers without thinking already, and Gemma has nothing to turn off.
func Thinks(model string) bool { return FamilyOf(model) != FamilyNone }

// RequestFields builds the fields that carry the thinking switch, for a host that serves
// the weights unchanged. It returns nil for a model with no switch, which is what
// openaicompat wants for "send the request as it stands".
//
// Thinking is off by default everywhere this is used, because reasoning spends the whole
// token budget and most of the latency before the first word of the answer, which is the
// wrong trade for a live conversation.
func RequestFields(model string, thinking bool) func(llm.ResponseParams, string) map[string]any {
	family := FamilyOf(model)
	if family == FamilyNone {
		return nil
	}

	return func(_ llm.ResponseParams, effort string) map[string]any {
		switch family {
		case FamilyGLM, FamilyMiniMax:
			return map[string]any{"thinking": map[string]any{"type": switchWord(thinking)}}
		case FamilyDeepSeekPro:
			fields := map[string]any{"thinking": map[string]any{"type": switchWord(thinking)}}
			if effort != "" {
				fields["reasoning_effort"] = effort
			}
			return fields
		case FamilyQwen:
			return map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": thinking}}
		default:
			return map[string]any{"chat_template_kwargs": map[string]any{"thinking": thinking}}
		}
	}
}

// Capabilities is what a model of this family accepts, for a host that serves the weights
// unchanged. Chat completions has nowhere to put a stored response, a conversation or a
// cache key, so openaicompat forces those off whatever this says.
func Capabilities(model string, thinking bool, effort string) llm.Capabilities {
	capabilities := llm.Capabilities{StreamsReasoning: thinking}
	if thinking {
		capabilities.ReasoningEfforts = Efforts(model)
		capabilities.DefaultEffort = effort
	}
	return capabilities
}

// switchWord is how a top-level thinking object spells on and off.
func switchWord(thinking bool) string {
	if thinking {
		return "enabled"
	}
	return "disabled"
}
