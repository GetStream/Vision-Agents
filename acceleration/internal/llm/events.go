package llm

import "time"

// Event is pulled from a Stream. The names follow the Responses API's own stream events.
type Event interface {
	isLLMEvent()
}

// ResponseCreated means the provider accepted a request and is working on it. It is the
// first event of every stream.
type ResponseCreated struct {
	ResponseID string
	Provider   string
	Model      string
	At         time.Time
}

func (ResponseCreated) isLLMEvent() {}

// OutputTextDelta is a piece of the answer. Index counts deltas within one response, so a
// consumer can tell order from arrival order.
//
// A delta does not say whether it is the last one, for the same reason a TTS audio chunk
// does not: waiting to find out would cost the latency the design is for.
// ResponseCompleted is what ends a response.
type OutputTextDelta struct {
	ResponseID string
	Index      int
	Delta      string
}

func (OutputTextDelta) isLLMEvent() {}

// ReasoningTextDelta is a piece of the model's thinking, which reasoning models stream
// before the answer itself. It is kept separate from OutputTextDelta because it must never
// be spoken or shown as the reply, but it is billed as output all the same.
type ReasoningTextDelta struct {
	ResponseID string
	Index      int
	Delta      string
}

func (ReasoningTextDelta) isLLMEvent() {}

// FunctionCallArgumentsDelta is a piece of a call the model is asking for. Index identifies
// which call within the response the piece belongs to, since a model may ask for several at
// once and providers interleave their fragments.
//
// Arguments arrive as JSON text a few characters at a time, so a delta is rarely parseable
// on its own. ResponseCompleted carries the assembled calls, which is what a caller that
// means to run one should wait for.
type FunctionCallArgumentsDelta struct {
	ResponseID string
	Index      int64
	// CallID and Name arrive on the first fragment of a call and are empty on the rest.
	CallID string
	Name   string
	Delta  string
}

func (FunctionCallArgumentsDelta) isLLMEvent() {}

// ResponseCompleted settles the stream and is always its last event, whatever ended it.
type ResponseCompleted struct {
	Response Response
}

func (ResponseCompleted) isLLMEvent() {}

// ResponseFailed reports a provider failure. It is followed by a ResponseCompleted with a
// failed status, so a caller counting responses still sees one end.
//
// Fatal means the provider cannot be used again.
type ResponseFailed struct {
	ResponseID string
	Provider   string
	Model      string
	Err        error
	Context    string
	Fatal      bool
}

func (f ResponseFailed) Error() string { return f.Err.Error() }

func (f ResponseFailed) Unwrap() error { return f.Err }

func (ResponseFailed) isLLMEvent() {}
