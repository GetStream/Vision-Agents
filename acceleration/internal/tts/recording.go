package tts

import (
	"context"
	"errors"
	"strings"
)

// Recording is a whole text to speak into one file, off the live path.
//
// It is a separate contract from TTS for the reason the transcription one is: a live voice
// is built around getting the first sound out early, and nothing is listening to an
// audiobook while it is being made. What matters instead is the file - a codec, a bitrate
// and one piece of audio rather than a stream of chunks.
type Recording struct {
	// Text is what to say, in whole paragraphs rather than the sentence at a time a
	// socket takes.
	Text string
	// Voice is the speaker, as the provider knows it.
	Voice string
	// Language is an ISO code, for the models that accept one.
	Language string
	// Format is the codec, sample rate and bitrate as one name - mp3_44100_128,
	// pcm_16000, ulaw_8000. Empty leaves the provider's default.
	Format string
	// Speed and Volume are relative to the voice's own, 1 being unchanged. Zero leaves
	// them alone.
	Speed  float64
	Volume float64
	// Emotion and Style are what to sound like, for the providers that take either.
	Emotion string
	Style   string
	// Stability is how much the voice may vary, and Similarity how closely a cloned
	// voice tracks its reference. Zero leaves the provider's defaults.
	Stability  float64
	Similarity float64
	// Pronunciations is how to say words the voice gets wrong, keyed by the word.
	Pronunciations map[string]string
}

// Validate reports whether there is anything to say.
func (r Recording) Validate() error {
	if strings.TrimSpace(r.Text) == "" {
		return errors.New("tts: there is nothing to say")
	}
	return nil
}

// Recorded is a whole text spoken.
type Recorded struct {
	// Audio is the file, encoded as Format says.
	Audio  []byte
	Format string
	// AudioDurationMs is how long it plays for, where the provider says. Zero when the
	// codec would have to be decoded to find out.
	AudioDurationMs int64
	// Characters is how much text was spoken, which is what it is billed on.
	Characters int64
}

// Recorder speaks a whole text into one file.
//
// Start and Close are what routing.Provider asks for rather than anything this needs: an
// HTTP client has nothing to open, and a provider missing its key fails when it is built,
// which is where the router picks the next candidate.
type Recorder interface {
	Record(ctx context.Context, recording Recording) (Recorded, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "elevenlabs".
	Provider() string
	// Model is the model identifier used in stats.
	Model() string
}
