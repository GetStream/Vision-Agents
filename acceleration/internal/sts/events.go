package sts

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Event is emitted on the channel returned by STS.Events.
type Event interface {
	isSTSEvent()
}

// Connected means the upstream session is open and the model has taken its configuration.
type Connected struct {
	Provider string
	Model    string
	At       time.Time
}

func (Connected) isSTSEvent() {}

// Disconnected means the upstream connection closed. Clean is true for a connection the
// vendor or the provider ended on purpose, which includes one that is about to be resumed.
type Disconnected struct {
	Provider string
	Model    string
	Reason   string
	Clean    bool
	At       time.Time
}

func (Disconnected) isSTSEvent() {}

// Error reports a provider failure. Fatal means the session cannot continue. ResponseID
// names the response the failure belongs to, when it belongs to one, so the response is
// still one stat row.
type Error struct {
	Provider   string
	Model      string
	ResponseID string
	Err        error
	Context    string
	Fatal      bool
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isSTSEvent() {}

// SpeechStarted means the model's own voice activity detector heard the participant
// begin. It says nothing about the response in flight: whether that was cut off is what
// ResponseComplete says, and only it.
type SpeechStarted struct {
	Participant Participant
	At          time.Time
}

func (SpeechStarted) isSTSEvent() {}

// SpeechStopped means the model judged the participant's turn over. It is the moment the
// caller starts waiting, and what a response's time to first byte is measured from.
type SpeechStopped struct {
	Participant Participant
	At          time.Time
}

func (SpeechStopped) isSTSEvent() {}

// InputTranscript is what the model heard the participant say. Mode says whether the text
// adds to, replaces or settles what came before, in the same words a transcriber uses.
type InputTranscript struct {
	Participant Participant
	Mode        stt.Mode
	Text        string
	Language    string
}

func (InputTranscript) isSTSEvent() {}

// OutputTranscript is what the model said, for the response it said it in.
type OutputTranscript struct {
	ResponseID string
	Mode       stt.Mode
	Text       string
}

func (OutputTranscript) isSTSEvent() {}

// ResponseStarted means the model began a reply. Generation counts replies from one, so a
// consumer can tell late audio from an earlier reply apart from the current one without
// keeping every response id it ever saw.
type ResponseStarted struct {
	ResponseID string
	Generation int
	At         time.Time
}

func (ResponseStarted) isSTSEvent() {}

// AudioChunk is a piece of the model's speech. Index counts chunks within one response, so
// a consumer can tell playback order from arrival order.
type AudioChunk struct {
	ResponseID string
	Generation int
	Index      int
	Audio      audio.PcmData
}

func (AudioChunk) isSTSEvent() {}

// Usage is what one response cost, where the model says. The audio tokens are a subset of
// the totals rather than an addition to them.
type Usage struct {
	InputTokens       int64
	CachedInputTokens int64
	OutputTokens      int64
	InputAudioTokens  int64
	OutputAudioTokens int64
}

// ResponseComplete settles one reply. It is the natural unit of billable work, the way
// SynthesisComplete is for a voice, and Interrupted true is the one signal that the caller
// cut in: a consumer that drops its buffered audio does so on this and nothing else.
type ResponseComplete struct {
	ResponseID string
	Generation int
	Provider   string
	Model      string
	// Interrupted is true when the caller spoke over the reply and the model stopped.
	Interrupted bool
	// AudioDurationMs is how much speech came back.
	AudioDurationMs float64
	// TimeToFirstByteMs is how long the caller waited, from the end of their turn to the
	// first audio of the reply. Zero for a reply that answered nobody's wait.
	TimeToFirstByteMs float64
	// ResponseTimeMs is the whole reply, first event to last.
	ResponseTimeMs float64
	Usage          Usage
}

func (ResponseComplete) isSTSEvent() {}

// ToolCall is the model asking for a tool to be run. The answer goes back through
// STS.Answer against the same CallID.
type ToolCall struct {
	ResponseID string
	CallID     string
	Name       string
	Arguments  string
}

func (ToolCall) isSTSEvent() {}

// ToolCancel means the model no longer wants the answers to those calls, usually because
// the caller interrupted the turn that asked.
type ToolCancel struct {
	CallIDs []string
}

func (ToolCancel) isSTSEvent() {}

// SessionExpiring warns that the vendor is about to cut the session off. A provider that
// can resume does so on its own; one that cannot lets this be the caller's warning.
type SessionExpiring struct {
	TimeLeft time.Duration
}

func (SessionExpiring) isSTSEvent() {}

// EmitterBuffer is the buffer a provider's emitter is given. It is deeper than a voice's
// because the model's audio arrives at a rate a slow consumer holds up, and a provider
// stalled on its emitter is late reading the frame that says the caller interrupted.
const EmitterBuffer = 256

// Emitter fans provider events out to a single consumer channel. Providers hold one
// rather than managing the channel and its close semantics themselves.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
