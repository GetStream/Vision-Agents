package llmrouter

import (
	"context"
	"net/http/httptrace"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stageTrace is when an attempt's HTTP request asked for a connection, got one, was written and
// was first answered. Its hooks run on the transport's goroutines. The stages do not add up to
// the time to first token: writing the request, the SDK's own steps and the choice of a key
// before it asks for a connection, and the backoff before a retry fall in none of them.
type stageTrace struct {
	mu     sync.Mutex
	stamps stamps
}

type stamps struct {
	getConn, gotConn, wrote, firstByte time.Time
	reused                             bool
}

// traceStages is ctx with hooks that stamp the attempt's request, after any trace the caller
// set, which httptrace keeps.
func traceStages(ctx context.Context) (context.Context, *stageTrace) {
	t := &stageTrace{}
	return httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
		GetConn: func(string) {
			t.mu.Lock()
			defer t.mu.Unlock()
			// A client that retries asks again: only the request that answered is the attempt's.
			t.stamps = stamps{getConn: time.Now()}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			t.mu.Lock()
			defer t.mu.Unlock()
			t.stamps.gotConn, t.stamps.reused = time.Now(), info.Reused
		},
		WroteRequest: func(info httptrace.WroteRequestInfo) {
			if info.Err != nil {
				return
			}
			t.mu.Lock()
			defer t.mu.Unlock()
			t.stamps.wrote = time.Now()
		},
		GotFirstResponseByte: func() {
			// Cancelling the request closes the connection, and the provider can still answer
			// it in that window: an answer the attempt gave up on is no wait.
			if ctx.Err() != nil {
				return
			}
			t.mu.Lock()
			defer t.mu.Unlock()
			t.stamps.firstByte = time.Now()
		},
	}), t
}

// observe records the stages the request passed by the time the provider's Create returned,
// and returns what it was stamped with. A provider that made no HTTP request has none.
func (t *stageTrace) observe(config routing.ProviderConfig) stamps {
	t.mu.Lock()
	passed := t.stamps
	t.mu.Unlock()
	if passed.gotConn.IsZero() {
		return passed
	}
	routing.CountLLMConnection(config.Provider, passed.reused)
	if !passed.reused {
		routing.ObserveLLMStage(config, routing.StageConnect, passed.gotConn.Sub(passed.getConn))
	}
	// A provider can answer, a 413 say, before the request is all written; that answer waited on
	// nothing.
	if !passed.wrote.IsZero() && passed.firstByte.After(passed.wrote) {
		routing.ObserveLLMStage(config, routing.StageWait, passed.firstByte.Sub(passed.wrote))
	}
	return passed
}

// observeStream records the stream stage of a settled response: from the first byte of the
// response to its first token, on the clock first output is measured on (the stream's own,
// which starts when Create returned). A response that never said anything has none.
func (s stamps) observeStream(config routing.ProviderConfig, createdAt time.Time, response llm.Response) {
	if s.firstByte.IsZero() || response.TimeToFirstTokenMs <= 0 {
		return
	}
	firstToken := createdAt.Add(time.Duration(response.TimeToFirstTokenMs * float64(time.Millisecond)))
	routing.ObserveLLMStage(config, routing.StageStream, firstToken.Sub(s.firstByte))
}
