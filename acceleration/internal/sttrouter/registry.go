package sttrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/grok"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/muse"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/parakeet"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/togetherparakeet"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[stt.STT]() }

// DefaultRegistry returns a registry with every speech-to-text provider this build
// supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(deepgram.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		options := deepgram.Options{
			Model:    spec.Model,
			Keyterms: spec.Keyterms,
			Logger:   spec.Logger,
		}
		// Flux decides where a turn ended itself, and eot_timeout_ms is how long a
		// silence has to be before it does, which is what a caller asking for silence
		// endpointing is asking for.
		if spec.STT.SilenceMs != nil {
			options.EotTimeoutMs = *spec.STT.SilenceMs
		}
		// Flux only accepts language hints on the multilingual model.
		if spec.Model == deepgram.MultilingualModel {
			options.LanguageHints = spec.LanguageHints
		}
		return deepgram.New(options)
	})

	registry.Register(gemini.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return gemini.New(gemini.Options{
			Model:         spec.Model,
			Keyterms:      spec.Keyterms,
			LanguageHints: spec.LanguageHints,
			Logger:        spec.Logger,
		})
	})

	registry.Register(parakeet.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return parakeet.New(parakeet.Options{Model: spec.Model, Logger: spec.Logger})
	})

	registry.Register(grok.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return grok.New(grok.Options{
			Model:    spec.Model,
			Keyterms: spec.Keyterms,
			Language: firstLanguage(spec.LanguageHints),
			// Always on, rather than only when a request asks: a request that asked would
			// be routed away from every provider that cannot diarize, which is four of the
			// six that serve a live call. The label is worth having where it is free and
			// not worth losing failover for. Muse is diarised by its own default.
			Diarize: true,
			Logger:  spec.Logger,
		})
	})

	registry.Register(muse.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		options := muse.Options{
			Model:         spec.Model,
			Keyterms:      spec.Keyterms,
			LanguageHints: spec.LanguageHints,
			Logger:        spec.Logger,
		}
		// Diarization on this model settles a turn about a second and a half later than
		// endpointing does, so unlike Grok it is what a request asks for rather than what
		// every call gets. Declared and honoured either way: a model that said it could
		// name the voice and then quietly did not would be worse than one that cannot.
		if spec.STT.Diarize != nil && *spec.STT.Diarize {
			options.Mode = muse.ModeDiarization
		}
		return muse.New(options)
	})

	registry.Register(togetherparakeet.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return togetherparakeet.New(togetherparakeet.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}

// firstLanguage picks the language to format the transcript for. xAI takes one code, and
// it only decides how numbers and currencies are written, so a list of hints has one
// useful answer in it.
func firstLanguage(hints []string) string {
	if len(hints) == 0 {
		return ""
	}
	return hints[0]
}
