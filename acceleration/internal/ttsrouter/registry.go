package ttsrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/breeze"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/fish"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/inworld"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/s2pro"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[tts.TTS]() }

// DefaultRegistry returns a registry with every text-to-speech provider this build
// supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(elevenlabs.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		settings := elevenlabsSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		options := elevenlabs.Options{
			Model:    spec.Model,
			VoiceID:  voiceOr(settings.VoiceID, spec.Voice),
			Language: firstLanguage(spec.LanguageHints),
			Logger:   spec.Logger,
		}
		// The v3 models are served on a different socket, so which endpoint to open is
		// decided by the model the router picked rather than by a second provider name.
		if elevenlabs.Performs(spec.Model) {
			return elevenlabs.NewDialogue(options)
		}
		return elevenlabs.New(options)
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

	registry.Register(breeze.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		return breeze.New(breeze.Options{Model: spec.Model, Voice: spec.Voice, Logger: spec.Logger})
	})

	registry.Register(inworld.ProviderName, func(spec routing.Spec) (tts.TTS, error) {
		settings := inworldSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		return inworld.New(inworld.Options{
			Model:        spec.Model,
			VoiceID:      voiceOr(settings.VoiceID, spec.Voice),
			DeliveryMode: settings.DeliveryMode,
			Logger:       spec.Logger,
		})
	})

	return registry
}

// elevenlabsSettings is what a caller can reach at ElevenLabs through overwrites.
//
// A voice id is here as well as in the shared vocabulary because it is the one setting
// that cannot be shared: an id from one vendor's library means nothing at another. A
// config that routes between vendors and wants a named voice at each says so once per
// vendor, and the one the router picks is the one that is read.
type elevenlabsSettings struct {
	VoiceID string `json:"voice_id"`
}

// inworldSettings is what a caller can reach at Inworld through overwrites. DeliveryMode
// is not in the shared vocabulary because it is three named modes rather than the
// continuous stability the others have, so there is nothing to standardise it against.
type inworldSettings struct {
	VoiceID      string `json:"voice_id"`
	DeliveryMode string `json:"delivery_mode"`
}

// voiceOr prefers a voice named for this vendor over the one the request asked of
// whoever answered. An overwrite is the more specific of the two: it was written knowing
// which vendor it is for.
func voiceOr(overwritten, asked string) string {
	if overwritten != "" {
		return overwritten
	}
	return asked
}

// firstLanguage picks the language to synthesise in. Speech is in one language at a time,
// so a list of hints only has one useful answer in it.
func firstLanguage(hints []string) string {
	if len(hints) == 0 {
		return ""
	}
	return hints[0]
}
