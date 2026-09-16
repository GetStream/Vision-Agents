package sttrouter

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/deepgram"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/elevenlabs"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/gemini"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/grok"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/inworld"
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
			// be routed away from every provider that cannot diarize, which is most of
			// those that serve a live call. The label is worth having where it is free and
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
		settings := museSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

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
		// A caller who wrote the mode out in Muse's own words meant that mode, so it wins
		// over what the shared terms worked out. muse.New refuses one it does not have.
		if settings.Mode != "" {
			options.Mode = settings.Mode
		}
		return muse.New(options)
	})

	registry.Register(cartesia.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		settings := cartesiaSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		options := cartesia.Options{
			Model:                 spec.Model,
			Keyterms:              spec.Keyterms,
			TurnStartThreshold:    settings.TurnStartThreshold,
			TurnEagerEndThreshold: settings.TurnEagerEndThreshold,
			TurnEndThreshold:      settings.TurnEndThreshold,
			Logger:                spec.Logger,
		}
		// Ink 2 decides where a turn ended itself, and turn_end_timeout_ms caps how long
		// it waits after the caller stops before it does, which is what a caller asking
		// for silence endpointing is asking for.
		if spec.STT.SilenceMs != nil {
			options.TurnEndTimeoutMs = *spec.STT.SilenceMs
		}
		if settings.TurnEndTimeoutMs != 0 {
			options.TurnEndTimeoutMs = settings.TurnEndTimeoutMs
		}
		return cartesia.New(options)
	})

	registry.Register(inworld.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		settings := inworldSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		options := inworld.Options{
			Model:                        spec.Model,
			Keyterms:                     spec.Keyterms,
			LanguageHints:                spec.LanguageHints,
			EndOfTurnConfidenceThreshold: settings.EndOfTurnConfidenceThreshold,
			VadThreshold:                 settings.VadThreshold,
			InactivityTimeoutSeconds:     settings.InactivityTimeoutSeconds,
			Logger:                       spec.Logger,
		}
		// The silence this model waits through once it is confident the turn is over.
		if spec.STT.SilenceMs != nil {
			options.MinEndOfTurnSilenceMs = *spec.STT.SilenceMs
		}
		if settings.MinEndOfTurnSilenceMs != 0 {
			options.MinEndOfTurnSilenceMs = settings.MinEndOfTurnSilenceMs
		}
		return inworld.New(options)
	})

	registry.Register(elevenlabs.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		settings := elevenlabsSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		options := elevenlabs.Options{
			Model:                spec.Model,
			Keyterms:             spec.Keyterms,
			LanguageHints:        spec.LanguageHints,
			VadThreshold:         settings.VadThreshold,
			MinSpeechDurationMs:  settings.MinSpeechDurationMs,
			MinSilenceDurationMs: settings.MinSilenceDurationMs,
			Logger:               spec.Logger,
		}
		// A segment here is settled by the voice activity detector, and its silence
		// threshold is the same question endpointing asks, in seconds rather than
		// milliseconds.
		if spec.STT.SilenceMs != nil {
			options.VadSilenceThresholdSecs = float64(*spec.STT.SilenceMs) / 1000
		}
		if settings.VadSilenceThresholdSecs != 0 {
			options.VadSilenceThresholdSecs = settings.VadSilenceThresholdSecs
		}
		return elevenlabs.New(options)
	})

	registry.Register(togetherparakeet.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		return togetherparakeet.New(togetherparakeet.Options{Model: spec.Model, Logger: spec.Logger})
	})

	// One factory for both Nemotron models: the English one and the multilingual one are
	// the same socket and the same protocol, and spec.Model picks between them.
	registry.Register(togethernemotron.ProviderName, func(spec routing.Spec) (stt.STT, error) {
		settings := togetherNemotronSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		return togethernemotron.New(togethernemotron.Options{
			Model:     spec.Model,
			TurnGrace: time.Duration(settings.TurnGraceMs) * time.Millisecond,
			Logger:    spec.Logger,
		})
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

// museSettings is the turn boundary as Muse states it: a mode rather than a threshold, and
// the only vocabulary this model has for one. DIARIZATION is also what the shared diarize
// term asks for, so a request can arrive at that mode either way; PUSH_TO_TALK is the one
// only an overwrite can ask for, and it hands the boundary back to the caller's own client.
type museSettings struct {
	Mode string `json:"mode"`
}

// togetherNemotronSettings is how long the transcript has to stop changing before the turn
// is treated as over.
//
// It is here rather than in the shared vocabulary because it is not the vendor's knob at
// all: nothing on this protocol says where a turn ended, so the wait is the router's own,
// and taking silence_ms for it would promise a server-side endpointer that does not exist.
type togetherNemotronSettings struct {
	TurnGraceMs int `json:"turn_grace_ms"`
}

// cartesiaSettings are Ink 2's turn-detection thresholds, which are the same idea as
// Flux's above and not the same numbers: Cartesia's own migration guide says the
// thresholds behave differently, so they are two vendors' knobs rather than one term.
type cartesiaSettings struct {
	TurnStartThreshold    float64 `json:"turn_start_threshold"`
	TurnEagerEndThreshold float64 `json:"turn_eager_end_threshold"`
	TurnEndThreshold      float64 `json:"turn_end_threshold"`
	TurnEndTimeoutMs      int     `json:"turn_end_timeout_ms"`
}

// inworldSettings are the turn detection Inworld's own model has.
//
// vad_threshold is a pointer because zero is a request rather than an unset field: it
// turns the server's turn detection off and leaves the boundaries to the client, which on
// this path means nothing would settle until the call ended.
type inworldSettings struct {
	EndOfTurnConfidenceThreshold float64  `json:"end_of_turn_confidence_threshold"`
	MinEndOfTurnSilenceMs        int      `json:"min_end_of_turn_silence_ms"`
	VadThreshold                 *float64 `json:"vad_threshold"`
	InactivityTimeoutSeconds     int      `json:"inactivity_timeout_seconds"`
}

// elevenlabsSettings are Scribe's voice activity detector, which is what decides where one
// committed segment ends and the next begins.
type elevenlabsSettings struct {
	VadThreshold            float64 `json:"vad_threshold"`
	VadSilenceThresholdSecs float64 `json:"vad_silence_threshold_secs"`
	MinSpeechDurationMs     int     `json:"min_speech_duration_ms"`
	MinSilenceDurationMs    int     `json:"min_silence_duration_ms"`
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
