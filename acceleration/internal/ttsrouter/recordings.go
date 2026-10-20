package ttsrouter

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/cartesia"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// Recorders is the set of batch voices a build can construct.
type Recorders = routing.Registry[tts.Recorder]

// batch is what this router's half of the provider list is, for a priority list that
// names a vendor rather than one of their models. A whole text wants the batch voice,
// which is the one that can be asked for a codec rather than raw audio.
var batch = false

// NewRecorderRegistry returns an empty registry.
func NewRecorderRegistry() *Recorders { return routing.NewRegistry[tts.Recorder]() }

// DefaultRecorderRegistry returns a registry with every batch voice this build supports.
func DefaultRecorderRegistry() *Recorders {
	registry := NewRecorderRegistry()

	registry.Register(elevenlabs.ProviderName, func(spec routing.Spec) (tts.Recorder, error) {
		settings := elevenlabsSettings{}
		if err := spec.Settings(&settings); err != nil {
			return nil, err
		}

		recorder, err := elevenlabs.NewRecorder(elevenlabs.RecorderOptions{
			VoiceID: voiceOr(settings.VoiceID, spec.Voice),
			Model:   spec.Model,
			Style:   settings.VoiceSettings.Style,
			// Checked against 0 to 1 here, before the call, as the live voices do.
			Stability:  spec.TTS.Stability,
			Similarity: spec.TTS.Similarity,
			Logger:     spec.Logger,
		})
		if err != nil {
			return nil, elevenlabsRefusal(spec.Model, err)
		}
		return recorder, nil
	})

	// Built from what the live Cartesia voice reads: the voice the router hands over (an
	// overwrite's voice_id among them), the model and the first language.
	registry.Register(cartesia.ProviderName, func(spec routing.Spec) (tts.Recorder, error) {
		recorder, err := cartesia.NewRecorder(cartesia.RecorderOptions{
			VoiceID:  spec.Voice,
			Model:    spec.Model,
			Language: firstLanguage(spec.LanguageHints),
			Format:   spec.TTS.Format,
			Logger:   spec.Logger,
		})
		// A format Cartesia cannot produce is the caller's to change, and says nothing
		// about the model's health.
		if errors.Is(err, cartesia.ErrFormat) {
			return nil, routing.Refused(err)
		}
		return recorder, err
	})

	return registry
}

// Recordings routes whole texts to a batch voice.
type Recordings struct {
	*routing.Router[tts.Recorder]
}

// NewRecordings validates the options and returns a router.
func NewRecordings(options Options) (*Recordings, error) {
	core, err := routing.New(routing.Options[tts.Recorder]{
		Modality: routing.TTS,
		Config:   options.Config,
		Registry: options.Recorders,
		Store:    options.Store,
		Live:     options.Live,
		Voices:   options.Voices,
		Gate:     options.Gate,
		Logger:   options.Logger,
	})
	if err != nil {
		return nil, err
	}
	return &Recordings{Router: core}, nil
}

// Recording is a whole text to speak and how to bill it.
type Recording struct {
	// CustomerID owns the request. It is what every statistic is keyed by.
	CustomerID string
	// Tags are the customer's own cost labels.
	Tags routing.Tags
	// Options are what the caller asked for, target and voice included.
	Options options.TTS
	// Text is what to say.
	Text string
}

// request is what a recording asks the router for.
func (recording Recording) request() routing.Request {
	return routing.Request{
		CustomerID:    recording.CustomerID,
		Tags:          recording.Tags,
		Target:        recording.Options.Target,
		Providers:     recording.Options.Providers,
		Realtime:      &batch,
		LanguageHints: recording.Options.Languages,
		Voice:         recording.Options.Voice,
		Terms:         recording.Options.Terms(),
		DataPolicy:    recording.Options.DataPolicy,
		TTS:           recording.Options,
	}
}

// Records reports whether a vendor's models can record here. Most vendors only stream.
func (r *Recordings) Records(provider string) bool { return r.Builds(provider) }

// Check refuses a recording that no candidate can record: a model of a vendor that only
// streams, or a shortcut or priority list made only of them. It names the models that
// can. One candidate that records is enough, since Record passes over the rest.
func (r *Recordings) Check(ctx context.Context, recording Recording) error {
	candidates, err := r.Candidates(ctx, recording.request())
	if err != nil {
		// Record refuses it in the words it always has.
		return nil
	}
	var streamed []string
	for _, candidate := range candidates {
		if r.Records(candidate.Config.Provider) {
			return nil
		}
		streamed = append(streamed, candidate.Config.Name())
	}
	var recorders []string
	for _, provider := range r.Config().Providers {
		if r.Records(provider.Provider) {
			recorders = append(recorders, provider.Name())
		}
	}
	return routing.Refused(stack.Wrap(fmt.Errorf("routing: %s can only stream speech, not record it; models that record: %s",
		strings.Join(streamed, ", "), strings.Join(recorders, ", "))))
}

// Record selects a voice and speaks the whole text, falling back to the next candidate
// when one fails to start.
func (r *Recordings) Record(
	ctx context.Context,
	recording Recording,
) (tts.Recorded, routing.ProviderConfig, error) {
	request := recording.request()

	provider, config, err := r.Select(ctx, request)
	if err != nil {
		return tts.Recorded{}, routing.ProviderConfig{}, err
	}
	defer provider.Close()

	startedAt := time.Now()
	// No voice is named here: the recorder was built for the voice this vendor knows the
	// request's by, or the one its overwrite named, and the request's own is neither.
	recorded, err := provider.Record(ctx, tts.Recording{
		Text:           recording.Text,
		Language:       first(recording.Options.Languages),
		Format:         recording.Options.Format,
		Speed:          number(recording.Options.Speed),
		Volume:         number(recording.Options.Volume),
		Emotion:        recording.Options.Emotion,
		Style:          recording.Options.Style,
		Stability:      number(recording.Options.Stability),
		Similarity:     number(recording.Options.Similarity),
		Pronunciations: recording.Options.Pronunciations,
	})

	stat := routing.Stat{
		Owner:     request.Owner(),
		StartedAt: startedAt.UTC(),
		LatencyMs: routing.MsSince(startedAt),
		Success:   err == nil,
		Usage: routing.Usage{
			Characters: recorded.Characters,
			AudioMs:    recorded.AudioDurationMs,
		},
	}
	if err != nil {
		stat.ErrorCode = "record_failed"
		stat.ErrorMessage = err.Error()
	}
	r.Recorder().Record(config, stat)
	if err != nil {
		return tts.Recorded{}, config, err
	}
	// Not the voice asked for: a recorder that cannot say leaves it empty rather than claim
	// a voice it may have dropped.
	if voiced, ok := provider.(tts.Voiced); ok {
		recorded.Voice = voiced.Voice()
	}
	return recorded, config, nil
}

// first is the one language a provider that takes a single code should be told about.
func first(languages []string) string {
	if len(languages) == 0 {
		return ""
	}
	return languages[0]
}

// number reads an optional setting, where unset means leave the voice alone.
func number(value *float64) float64 {
	if value == nil {
		return 0
	}
	return *value
}
