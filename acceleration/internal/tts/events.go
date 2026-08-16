package tts

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
)

// Event is emitted on the channel returned by TTS.Events.
type Event interface {
	isTTSEvent()
}

// SynthesisStarted means the provider accepted an utterance and is working on it.
type SynthesisStarted struct {
	SynthesisID string
	Provider    string
	Model       string
	Voice       string
	At          time.Time
}

func (SynthesisStarted) isTTSEvent() {}

// AudioChunk is a piece of synthesised speech. Index counts chunks within one synthesis,
// so a consumer can tell playback order from arrival order.
//
// A chunk does not say whether it is the last one: a streaming provider only learns that
// after the fact, and buffering a chunk to find out would cost the latency the whole
// design is for. SynthesisComplete is what ends an utterance.
type AudioChunk struct {
	SynthesisID string
	Index       int
	Audio       audio.PcmData
}

func (AudioChunk) isTTSEvent() {}

// SynthesisComplete settles one utterance. It carries everything a stat row needs, since
// this is the natural unit of billable work.
type SynthesisComplete struct {
	SynthesisID string
	Provider    string
	Model       string
	// Characters is the text that was billed for.
	Characters int64
	// AudioDurationMs is how much speech came back.
	AudioDurationMs float64
	// TimeToFirstByteMs is how long the listener waited to hear anything, which is the
	// number that decides whether a voice agent feels alive.
	TimeToFirstByteMs float64
	// SynthesisTimeMs is the whole utterance, first request to last chunk.
	SynthesisTimeMs float64
	// Interrupted is true when barge-in cut the utterance short.
	Interrupted bool
}

func (SynthesisComplete) isTTSEvent() {}

// Connected means the upstream connection is established.
type Connected struct {
	Provider string
	Model    string
	At       time.Time
}

func (Connected) isTTSEvent() {}

// Disconnected means the upstream connection closed. Clean is false for failures.
type Disconnected struct {
	Provider string
	Model    string
	Reason   string
	Clean    bool
	At       time.Time
}

func (Disconnected) isTTSEvent() {}

// Error reports a provider failure. Fatal means the session cannot continue.
type Error struct {
	Provider    string
	Model       string
	SynthesisID string
	Err         error
	Context     string
	Fatal       bool
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isTTSEvent() {}

// Emitter fans provider events out to a single consumer channel. Providers hold one
// rather than managing the channel and its close semantics themselves.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
