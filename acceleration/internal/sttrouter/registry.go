package sttrouter

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/grok"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/muse"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/parakeet"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/togethernemotron"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/togetherparakeet"
)

// NewRegistry returns an empty registry.
func NewRegistry() *Registry { return routing.NewRegistry[stt.STT]() }

// DefaultRegistry returns a registry with every speech-to-text provider this build
// supports.
func DefaultRegistry() *Registry {
	registry := NewRegistry()

	registry.Register(deepgram.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		settings := deepgramSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		options := deepgram.Options{
			Model:             spec.Model,
			Keyterms:          spec.Keyterms,
			MipOptOut:         trainingRefused(spec),
			EotThreshold:      settings.EotThreshold,
			EagerEotThreshold: settings.EagerEotThreshold,
			Logger:            spec.Logger,
		}
		// Flux decides where a turn ended itself, and eot_timeout_ms is how long a
		// silence has to be before it does, which is what a caller asking for silence
		// endpointing is asking for.
		if spec.STT.SilenceMs != nil {
			options.EotTimeoutMs = *spec.STT.SilenceMs
		}
		if settings.EotTimeoutMs != 0 {
			options.EotTimeoutMs = settings.EotTimeoutMs
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
			Mode:          transcriptionMode(spec.STT.Mode),
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
			// Verbatim here is only the fillers, which is why this model declares that
			// half of Mode and not the other: it keeps the ums, it does not rewrite the
			// sentence they were in.
			FillerWords: spec.STT.Mode == options.ModeVerbatim,
			Logger:      spec.Logger,
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

	// One factory for both Nemotron models: the English one and the multilingual one are
	// the same socket and the same protocol, and spec.Model picks between them.
	registry.Register(togethernemotron.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return togethernemotron.New(togethernemotron.Options{Model: spec.Model, Logger: spec.Logger})
	})

	return registry
}

// deepgramSettings are the Flux turn-detection thresholds a caller can reach through
// overwrites. They are not in the shared vocabulary because nobody else has them: an
// eot_threshold is a confidence that this model's own turn detector has decided, and
// there is no second provider to standardise it against.
type deepgramSettings struct {
	EotThreshold      float64 `json:"eot_threshold"`
	EagerEotThreshold float64 `json:"eager_eot_threshold"`
	EotTimeoutMs      int     `json:"eot_timeout_ms"`
}

// trainingRefused reports whether this request asked not to be trained on.
//
// Deepgram takes it per request rather than per account, so unlike the retention half of
// the same policy it is something to send. It is sent whenever the caller asked, and the
// declaration in router.yaml is what promises the request carries it.
func trainingRefused(spec routing.Spec) bool {
	allowed := spec.STT.DataPolicy.AllowTraining
	return allowed != nil && !*allowed
}

// transcriptionMode is what Gemini calls the mode this request asked for. Empty leaves it
// to the server, whose own default is verbatim.
func transcriptionMode(mode string) gemini.TranscriptionMode {
	switch mode {
	case options.ModeVerbatim:
		return gemini.ModeVerbatim
	case options.ModeSmart:
		return gemini.ModeSmart
	default:
		return ""
	}
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
