package ttsrouter

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/elevenlabs"
)

// Recorders is the set of batch voices a build can construct.
type Recorders = routing.Registry[tts.Recorder]

// NewRecorderRegistry returns an empty registry.
func NewRecorderRegistry() *Recorders { return routing.NewRegistry[tts.Recorder]() }

// DefaultRecorderRegistry returns a registry with every batch voice this build supports.
func DefaultRecorderRegistry() *Recorders {
	registry := NewRecorderRegistry()

	registry.Register(elevenlabs.ProviderName, func(spec routing.Spec) (tts.Recorder, error) {
		return elevenlabs.NewRecorder(elevenlabs.RecorderOptions{
			VoiceID: spec.Voice,
			Model:   spec.Model,
			Logger:  spec.Logger,
		})
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

// Record selects a voice and speaks the whole text, falling back to the next candidate
// when one fails to start.
func (r *Recordings) Record(
	ctx context.Context,
	recording Recording,
) (tts.Recorded, routing.ProviderConfig, error) {
	request := routing.Request{
		CustomerID:    recording.CustomerID,
		Tags:          recording.Tags,
		Target:        recording.Options.Target,
		LanguageHints: recording.Options.Languages,
		Voice:         recording.Options.Voice,
		Terms:         recording.Options.Terms(),
		TTS:           recording.Options,
	}

	provider, config, err := r.Select(ctx, request)
	if err != nil {
		return tts.Recorded{}, routing.ProviderConfig{}, err
	}
	defer provider.Close()

	startedAt := time.Now()
	recorded, err := provider.Record(ctx, tts.Recording{
		Text:           recording.Text,
		Voice:          recording.Options.Voice,
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
	}
	r.Recorder().Record(config, stat)
	if err != nil {
		return tts.Recorded{}, config, err
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
