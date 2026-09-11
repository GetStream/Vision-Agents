// Package sts defines the minimal speech-to-speech contract shared by every native audio
// model: the caller's audio and, where the model sees, frames in; audio, transcripts and
// turn boundaries out.
//
// The model owns endpointing, transcription, synthesis and barge-in detection. A running
// session is the voice activity detector, the turn detector, the transcriber and the voice
// all at once, and nothing in front of it may act as one of them. Only the pieces the
// router needs are standardised here; anything provider-specific stays on the concrete
// type.
package sts

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// InputSampleRate is the rate every provider is fed, which is what the call edge and the
// transcribers already produce. A model that wants another rate resamples on its own side
// rather than asking the edge to change.
const InputSampleRate = stt.SampleRate

// PcmData is a chunk of signed 16-bit PCM audio.
type PcmData = audio.PcmData

// Participant identifies who is speaking, and is what a transcript is attributed to.
type Participant = stt.Participant

// These are what a provider answers with for what its model cannot do. They are refusals
// rather than silences: a typed turn that was dropped would leave the caller believing it
// had been heard.
var (
	ErrNoText            = errors.New("sts: this model does not take typed turns")
	ErrNoImages          = errors.New("sts: this model does not accept images")
	ErrNoTools           = errors.New("sts: this model does not call tools")
	ErrInstructionsFixed = errors.New("sts: this model takes its instructions only when the session opens")
	ErrToolsFixed        = errors.New("sts: this model takes its tools only when the session opens")
)

// Capabilities is what one model can be asked for.
//
// It is static per model rather than learned from the session, because routing checks a
// model's declaration in config against it before a session is opened, and a config that
// promised a term the code behind the model cannot send has to fail then rather than on a
// call.
type Capabilities struct {
	// InputModalities are the input kinds this model accepts besides audio, e.g. "image".
	InputModalities []string
	// Text reports whether a typed turn can be injected into the conversation.
	Text bool
	// Tools reports whether the model calls functions.
	Tools bool
	// InputTranscript and OutputTranscript report whether the model writes down what it
	// heard and what it said.
	InputTranscript  bool
	OutputTranscript bool
	// SemanticTurns is a turn detector that reads the words, ManualTurns one the caller
	// drives, and Endpointing a silence timer whose threshold can be set.
	SemanticTurns bool
	ManualTurns   bool
	Endpointing   bool
	// InstructionsMidSession and ToolsMidSession report whether either can change once
	// the session is open. Where they cannot, the matching method returns an error.
	InstructionsMidSession bool
	ToolsMidSession        bool
	// Usage reports whether the model says what a response cost in tokens.
	Usage bool
	// Resumable reports whether a session survives its connection, so the provider can
	// reconnect without the conversation starting over.
	Resumable bool
	// MaxDuration is how long the vendor lets one session run. Zero means the vendor has
	// published no limit, which is not the same as there being none.
	MaxDuration time.Duration
}

// Accepts reports whether this model takes that input kind. Audio and text are implicit.
func (c Capabilities) Accepts(modality string) bool {
	if modality == "" || modality == "text" || modality == "audio" {
		return true
	}
	return slices.Contains(c.InputModalities, modality)
}

// Expresses reports whether this model can honour a term a request may name. A term this
// modality has no word for is refused, so a config cannot declare one by mistake.
func (c Capabilities) Expresses(term options.Term) bool {
	switch term {
	case options.SemanticTurns:
		return c.SemanticTurns
	case options.ManualTurns:
		return c.ManualTurns
	case options.Endpointing:
		return c.Endpointing
	case options.Tools:
		return c.Tools
	case options.TextInput:
		return c.Text
	case options.InputTranscript:
		return c.InputTranscript
	case options.OutputTranscript:
		return c.OutputTranscript
	default:
		return false
	}
}

// STS is one live speech-to-speech conversation.
//
// Start opens the upstream session and returns once the model has acknowledged its
// configuration, so a caller told the session is ready is not told early. ProcessAudio
// feeds it, and Events carries audio, transcripts and turn boundaries back. Events is
// closed by Close.
type STS interface {
	Start(ctx context.Context) error
	// ProcessAudio streams one chunk of the caller's speech, mono at InputSampleRate.
	ProcessAudio(pcm PcmData, participant Participant) error
	// SendText injects a turn as though the participant had typed it. A model that takes
	// no typed turns returns ErrNoText rather than dropping it.
	SendText(text string, participant Participant) error
	// SendFrame offers a still image to a model that sees. One that does not returns
	// ErrNoImages.
	SendFrame(frame llm.ImagePart) error
	// SetInstructions changes the system prompt mid-call, on the models that allow it.
	SetInstructions(text string) error
	// SetTools replaces the tools the model may call, on the models that allow it.
	SetTools(tools []llm.Tool) error
	// Answer returns what a tool produced, or the error it failed with, against the call
	// that asked for it. A model whose wire needs a nudge to carry on after a tool result
	// gets it here, so the caller need not know which do.
	Answer(callID string, output string, err error) error
	// Prompt asks the model to respond now, guided by the text. It is not verbatim
	// speech: the model says what it makes of the text, which is what a greeting or a
	// "speak when idle" needs and what a caller wanting exact words must not be given.
	Prompt(text string) error
	// Interrupt stops the response in flight. playedMs is how much of it the listener
	// heard, so a model that keeps the conversation can be told where it was cut off;
	// zero means unknown, and the provider uses what it has sent. It is not an error to
	// interrupt silence.
	Interrupt(playedMs int) error
	Events() <-chan Event
	Close() error

	// Provider is the stable provider name used in stats, e.g. "openai".
	Provider() string
	// Model is the model identifier used in stats, e.g. "gpt-realtime-2".
	Model() string
	// SampleRate is the rate the audio on Events arrives at.
	SampleRate() int
	// Capabilities is what this model can be asked for.
	Capabilities() Capabilities
}
