package stsrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/openairealtime"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[sts.STS]() }

// DefaultRegistry returns a registry with every speech-to-speech provider this build
// supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	// Three vendors, one package: xAI and Alibaba speak OpenAI's events, and differ from
	// it only in how a session is configured, which the vendor value says.
	registry.Register(openairealtime.OpenAI.Provider, realtime(openairealtime.OpenAI))
	registry.Register(openairealtime.XAI.Provider, realtime(openairealtime.XAI))
	registry.Register(openairealtime.Qwen.Provider, realtime(openairealtime.Qwen))

	registry.Register(gemini.ProviderName, func(spec routing.Spec) (sts.STS, error) {
		settings := geminiSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}
		return gemini.New(gemini.Options{
			Model:            spec.Model,
			Voice:            spec.Voice,
			Instructions:     spec.STS.Instructions,
			Tools:            spec.Tools,
			SilenceMs:        spec.STS.SilenceMs,
			PrefixPaddingMs:  spec.STS.PrefixPaddingMs,
			StartSensitivity: settings.StartSensitivity,
			EndSensitivity:   settings.EndSensitivity,
			ThinkingLevel:    settings.ThinkingLevel,
			ProactiveAudio:   settings.ProactiveAudio,
			AffectiveDialog:  settings.AffectiveDialog,
			MediaResolution:  settings.MediaResolution,
			InputTranscript:  on(spec.STS.InputTranscript),
			OutputTranscript: on(spec.STS.OutputTranscript),
			Logger:           spec.Logger,
		})
	})

	return registry
}

// realtime builds the factory for one vendor reached over the OpenAI realtime protocol.
func realtime(vendor openairealtime.Vendor) routing.Factory[sts.STS] {
	return func(spec routing.Spec) (sts.STS, error) {
		settings := realtimeSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}
		return openairealtime.New(openairealtime.Options{
			Vendor:             vendor,
			Model:              spec.Model,
			Voice:              spec.Voice,
			Instructions:       spec.STS.Instructions,
			Tools:              spec.Tools,
			TurnDetection:      spec.STS.TurnDetection,
			SilenceMs:          spec.STS.SilenceMs,
			PrefixPaddingMs:    spec.STS.PrefixPaddingMs,
			Threshold:          settings.Threshold,
			Eagerness:          settings.Eagerness,
			InterruptResponse:  spec.STS.InterruptResponse,
			InputTranscript:    on(spec.STS.InputTranscript),
			TranscriptionModel: settings.TranscriptionModel,
			Logger:             spec.Logger,
		})
	}
}

// capabilitiesFor is what a model at a provider this build ships can be asked for, from the
// providers' own tables. It is how New checks the config before any session exists, and it
// reports false for a provider it does not know by name.
func capabilitiesFor(provider, model string) (sts.Capabilities, bool) {
	switch provider {
	case openairealtime.OpenAI.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.OpenAI, model), true
	case openairealtime.XAI.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.XAI, model), true
	case openairealtime.Qwen.Provider:
		return openairealtime.CapabilitiesFor(openairealtime.Qwen, model), true
	case gemini.ProviderName:
		return gemini.CapabilitiesFor(model), true
	default:
		return sts.Capabilities{}, false
	}
}

// realtimeSettings are what a caller can reach at the OpenAI-protocol vendors through
// overwrites. A threshold is the silence timer's own activation level, which only this
// family takes as a number; eagerness is how quickly the semantic detector decides, which
// only OpenAI has; the transcription model is the transcriber's name where one can be
// chosen. None of them is in the shared vocabulary because there is no second family to
// standardise them against.
type realtimeSettings struct {
	Threshold          *float64 `json:"threshold"`
	Eagerness          string   `json:"eagerness"`
	TranscriptionModel string   `json:"transcription_model"`
}

// geminiSettings are what a caller can reach at Google through overwrites: the detector's
// named sensitivities in place of a threshold, how long a 3.1 model thinks, the 2.5 models'
// proactive and affective modes, and how closely the model looks at a frame.
type geminiSettings struct {
	StartSensitivity string `json:"start_sensitivity"`
	EndSensitivity   string `json:"end_sensitivity"`
	ThinkingLevel    string `json:"thinking_level"`
	ProactiveAudio   *bool  `json:"proactive_audio"`
	AffectiveDialog  *bool  `json:"affective_dialog"`
	MediaResolution  string `json:"media_resolution"`
}

func on(flag *bool) bool { return flag != nil && *flag }
