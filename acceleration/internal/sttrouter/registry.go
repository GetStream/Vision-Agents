package sttrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/parakeet"
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

	return registry
}
