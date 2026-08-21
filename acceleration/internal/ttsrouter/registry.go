package ttsrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/fish"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/s2pro"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[tts.TTS]() }

// DefaultRegistry returns a registry with every text-to-speech provider this build
// supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(elevenlabs.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		return elevenlabs.New(elevenlabs.Options{
			Model:    spec.Model,
			VoiceID:  spec.Voice,
			Language: firstLanguage(spec.LanguageHints),
			Logger:   spec.Logger,
		})
	})

	registry.Register(cartesia.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		return cartesia.New(cartesia.Options{
			Model:    spec.Model,
			VoiceID:  spec.Voice,
			Language: firstLanguage(spec.LanguageHints),
			Logger:   spec.Logger,
		})
	})

	registry.Register(fish.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		return fish.New(fish.Options{Model: spec.Model, Voice: spec.Voice, Logger: spec.Logger})
	})

	registry.Register(s2pro.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		return s2pro.New(s2pro.Options{Model: spec.Model, Voice: spec.Voice, Logger: spec.Logger})
	})

	return registry
}

// firstLanguage picks the language to synthesise in. Speech is in one language at a time,
// so a list of hints only has one useful answer in it.
func firstLanguage(hints []string) string {
	if len(hints) == 0 {
		return ""
	}
	return hints[0]
}
