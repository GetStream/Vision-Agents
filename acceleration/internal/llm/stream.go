package llm

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Puller advances a provider's upstream stream by one chunk.
//
// It exists so a Stream needs no goroutine of its own: Next pulls, which means there is
// nothing to leak when a caller walks away from a response part-way through, and no
// channel whose close semantics have to be got right in every provider.
type Puller interface {
	// Advance reads the next chunk and records what it carried on the writer. It returns
	// false when the response has ended, at which point Err reports why and the upstream
	// has been released. It is called only from the goroutine calling Next.
	Advance(w *ResponseWriter) bool
	// Err is the failure that ended the stream, if there was one. A cancelled upstream is
	// not one: closing a stream is how a response is abandoned.
	Err() error
	// Close abandons the upstream. It is safe to call from any goroutine and may run
	// while Advance is blocked, which is what makes barge-in work.
	Close() error
}

// Stream is the response one Create produced.
//
// It is pulled rather than pushed, in the manner of the OpenAI SDK's own streams: Next
// advances it, Current is the event it advanced to, and Response is the whole answer once
// Next has returned false. A Stream is used from one goroutine; only Close may be called
// from another.
type Stream struct {
	writer *ResponseWriter
	puller Puller

	current  Event
	response Response
	err      error
	ended    bool
	watchers []func(Event)

	// closed is read by the goroutine calling Next and written by whichever one closes
	// the stream, which is the whole of the concurrency here.
	closed    atomic.Bool
	closeOnce sync.Once
	closeErr  error
}

// StreamOptions identifies the response a Stream carries.
type StreamOptions struct {
	// ResponseID correlates the events. One is generated when the caller has none.
	ResponseID string
	Provider   string
	Model      string
}

// NewStream returns the stream a provider produces by advancing the given puller.
func NewStream(options StreamOptions, puller Puller) *Stream {
	if options.ResponseID == "" {
		options.ResponseID = fmt.Sprintf("resp-%d", time.Now().UnixNano())
	}

	stream := &Stream{puller: puller}
	stream.writer = &ResponseWriter{
		stream:    stream,
		options:   options,
		startedAt: time.Now(),
		status:    StatusCompleted,
		calls:     map[int64]*partialCall{},
	}
	stream.writer.emit(ResponseCreated{
		ResponseID: options.ResponseID,
		Provider:   options.Provider,
		Model:      options.Model,
		At:         stream.writer.startedAt,
	})
	return stream
}

// Replay returns a Stream over events that have already happened, for a caller standing in
// for a provider.
func Replay(events ...Event) *Stream {
	stream := &Stream{ended: true}
	stream.writer = &ResponseWriter{stream: stream, pending: events}
	return stream
}

// Next advances to the next event, returning false once the response has settled. The last
// event of every stream is a ResponseCompleted, whatever ended it.
func (s *Stream) Next() bool {
	for {
		if len(s.writer.pending) > 0 {
			s.current = s.writer.pending[0]
			s.writer.pending = s.writer.pending[1:]
			s.record(s.current)
			return true
		}
		if s.ended {
			return false
		}
		if !s.puller.Advance(s.writer) {
			s.ended = true
			// A stream the caller closed was abandoned rather than finished, however the
			// provider happened to notice.
			if s.closed.Load() {
				s.writer.Cancelled()
			}
			s.writer.settle(s.puller.Err())
		}
	}
}

// Current is the event Next advanced to.
func (s *Stream) Current() Event { return s.current }

// Err is the provider failure that ended the stream, or nil. A stream that was closed
// part-way through did not fail: it was abandoned.
func (s *Stream) Err() error { return s.err }

// Response is the whole answer. It is settled once Next has returned false, and reports
// what had arrived so far before then.
func (s *Stream) Response() Response { return s.response }

// Close abandons the response. It is how barge-in stops a model mid-sentence, and it is
// safe to call from another goroutine and more than once.
//
// The stream still has to be drained afterwards: a response that was cut short still
// generated tokens, and it is the ResponseCompleted at the end that reports them.
func (s *Stream) Close() error {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		if s.puller != nil {
			s.closeErr = s.puller.Close()
		}
	})
	return s.closeErr
}

// Observe adds a function that sees every event as the stream is drained, for a caller
// that has to know what a response cost without standing between the events and whoever
// asked for them. It is what the router records its statistics from.
func (s *Stream) Observe(watch func(Event)) *Stream {
	s.watchers = append(s.watchers, watch)
	return s
}

// record keeps the stream's own view of what has passed and lets the watchers see it.
func (s *Stream) record(event Event) {
	switch typed := event.(type) {
	case ResponseCompleted:
		s.response = typed.Response
	case ResponseFailed:
		s.err = typed
	}
	for _, watch := range s.watchers {
		watch(event)
	}
}

// Collect drains a stream and returns the whole answer. It is for the model calls nobody
// is streaming: a reviewer summarising a call, a judge ruling on one, a model asked to
// rewrite a scenario.
func Collect(stream *Stream) (Response, error) {
	defer stream.Close()

	for stream.Next() {
	}
	if err := stream.Err(); err != nil {
		return stream.Response(), err
	}
	response := stream.Response()
	if response.Status == StatusCancelled {
		return response, errors.New("llm: the response was abandoned before it finished")
	}
	return response, nil
}

// ResponseWriter is how a provider records what came back. A provider is handed one by
// Puller.Advance and never holds it itself.
//
// Every method both records what it was told and queues the event a consumer sees for it,
// so the running answer and the deltas cannot drift apart.
type ResponseWriter struct {
	stream  *Stream
	options StreamOptions
	pending []Event

	startedAt          time.Time
	firstTokenAt       time.Time
	text               strings.Builder
	deltas             int
	usage              Usage
	status             ResponseStatus
	incompleteReason   string
	providerResponseID string
	settled            bool

	// calls holds the calls being assembled, keyed by the index the provider streams them
	// under, and order remembers which index arrived first so the calls settle in the
	// order the model asked for them.
	calls map[int64]*partialCall
	order []int64
}

// partialCall is one tool call being assembled from the fragments a provider streams.
type partialCall struct {
	id        string
	name      string
	arguments strings.Builder
	signature string
}

// ResponseID is the id the events of this response carry, which is the caller's.
func (w *ResponseWriter) ResponseID() string { return w.options.ResponseID }

// SetProviderResponseID records what the provider stored the response as, so a later
// response can continue from it. It never changes the id the events carry: the caller
// correlates a turn by the id it asked under.
func (w *ResponseWriter) SetProviderResponseID(id string) {
	if id == "" {
		return
	}
	w.providerResponseID = id
}

// OutputText records a piece of the answer.
func (w *ResponseWriter) OutputText(delta string) {
	if delta == "" {
		return
	}
	w.markFirstToken()
	w.text.WriteString(delta)
	w.emit(OutputTextDelta{ResponseID: w.options.ResponseID, Index: w.next(), Delta: delta})
}

// ReasoningText records a piece of the model's thinking. Thinking counts towards time to
// first token, since it is the provider working, but it is not part of the answer.
func (w *ResponseWriter) ReasoningText(delta string) {
	if delta == "" {
		return
	}
	w.markFirstToken()
	w.emit(ReasoningTextDelta{ResponseID: w.options.ResponseID, Index: w.next(), Delta: delta})
}

// FunctionCall records a fragment of a call the model is asking for.
//
// A call arrives in pieces under one index: the first carries the id and the name, and the
// arguments follow as JSON text spread over the fragments after it. Anything empty leaves
// what was already recorded alone, so a fragment carrying only arguments does not erase the
// name, and a signature need not arrive on the fragment that carried the name.
func (w *ResponseWriter) FunctionCall(index int64, callID, name, arguments, signature string) {
	w.markFirstToken()

	call, known := w.calls[index]
	if !known {
		call = &partialCall{}
		w.calls[index] = call
		w.order = append(w.order, index)
	}
	if callID != "" {
		call.id = callID
	}
	if name != "" {
		call.name = name
	}
	if signature != "" {
		call.signature = signature
	}
	call.arguments.WriteString(arguments)

	w.emit(FunctionCallArgumentsDelta{
		ResponseID: w.options.ResponseID,
		Index:      index,
		CallID:     callID,
		Name:       name,
		Delta:      arguments,
	})
}

// Usage records what the provider reported. Providers that repeat a cumulative usage frame
// on every chunk can call it as often as they like; the last one wins.
func (w *ResponseWriter) Usage(usage Usage) {
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	}
	w.usage = usage
}

// Incomplete records that the model stopped for a reason of the caller's making, such as
// the token cap.
func (w *ResponseWriter) Incomplete(reason string) {
	w.status = StatusIncomplete
	w.incompleteReason = reason
}

// Cancelled records that the response was abandoned before the model finished, which on
// the live path is barge-in. What did arrive still counts and is still billed.
func (w *ResponseWriter) Cancelled() {
	w.status = StatusCancelled
	w.incompleteReason = ""
}

// Fail reports a provider failure. The stream still settles afterwards, so a caller
// counting responses sees this one end.
func (w *ResponseWriter) Fail(err error, context string, fatal bool) {
	w.status = StatusFailed
	w.emit(ResponseFailed{
		ResponseID: w.options.ResponseID,
		Provider:   w.options.Provider,
		Model:      w.options.Model,
		Err:        err,
		Context:    context,
		Fatal:      fatal,
	})
}

// Text is the answer so far, reasoning excluded.
func (w *ResponseWriter) Text() string { return w.text.String() }

// settle queues the event that ends the stream. A response that produced nothing reports a
// zero time to first token rather than the whole duration.
func (w *ResponseWriter) settle(failure error) {
	if w.settled {
		return
	}
	w.settled = true

	if failure != nil && w.status != StatusFailed && w.status != StatusCancelled {
		w.Fail(failure, "stream", false)
	}

	var timeToFirstToken float64
	if !w.firstTokenAt.IsZero() {
		timeToFirstToken = float64(w.firstTokenAt.Sub(w.startedAt).Microseconds()) / 1000
	}

	w.emit(ResponseCompleted{Response: Response{
		ID:                 w.options.ResponseID,
		ProviderResponseID: w.providerResponseID,
		Provider:           w.options.Provider,
		Model:              w.options.Model,
		OutputText:         w.text.String(),
		ToolCalls:          w.assembled(),
		Usage:              w.usage,
		Status:             w.status,
		IncompleteReason:   w.incompleteReason,
		TimeToFirstTokenMs: timeToFirstToken,
		DurationMs:         float64(time.Since(w.startedAt).Microseconds()) / 1000,
	}})
}

// assembled returns the tool calls in the order the model asked for them.
//
// A provider that names its calls but does not identify them gets an id made here, because
// the result sent back has to say which call it answers and the model gave nothing else to
// say it with.
func (w *ResponseWriter) assembled() []ToolCall {
	if len(w.order) == 0 {
		return nil
	}

	calls := make([]ToolCall, 0, len(w.order))
	for _, index := range w.order {
		call := w.calls[index]
		id := call.id
		if id == "" {
			id = fmt.Sprintf("%s-tool-%d", w.options.ResponseID, index)
		}
		calls = append(calls, ToolCall{
			ID:        id,
			Name:      call.name,
			Arguments: call.arguments.String(),
			Signature: call.signature,
		})
	}
	return calls
}

func (w *ResponseWriter) emit(event Event) { w.pending = append(w.pending, event) }

// next numbers a delta. Reasoning and text share one sequence, so order is total.
func (w *ResponseWriter) next() int {
	index := w.deltas
	w.deltas++
	return index
}

// markFirstToken stamps the first sign of life from the provider.
func (w *ResponseWriter) markFirstToken() {
	if w.firstTokenAt.IsZero() {
		w.firstTokenAt = time.Now()
	}
}
