package stt

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
)

// Event is emitted on the channel returned by STT.Events.
type Event interface {
	isSTTEvent()
}

// Transcript carries recognised text. Mode says whether it supersedes the previous one.
type Transcript struct {
	Participant      Participant
	Mode             Mode
	Text             string
	Confidence       float64
	Language         string
	Provider         string
	Model            string
	ProcessingTimeMs float64
	AudioDurationMs  float64
}

// Final reports whether the turn is settled.
func (t Transcript) Final() bool { return t.Mode == ModeFinal }

func (Transcript) isSTTEvent() {}

// Connected means the upstream connection is established.
type Connected struct {
	Provider string
	Model    string
	At       time.Time
}

func (Connected) isSTTEvent() {}

// Disconnected means the upstream connection closed. Clean is false for failures.
type Disconnected struct {
	Provider string
	Model    string
	Reason   string
	Clean    bool
	At       time.Time
}

func (Disconnected) isSTTEvent() {}

// Error reports a provider failure. Fatal means the session cannot continue.
type Error struct {
	Provider string
	Model    string
	Err      error
	Context  string
	Fatal    bool
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isSTTEvent() {}

// Emitter fans provider events out to a single consumer channel. Providers hold one
// rather than managing the channel and its close semantics themselves.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
