package llmrouter

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

var tracer = tracing.Tracer("llmrouter")

// statRecorder is where a session writes the stat row of each response. It is the router's
// recorder, and a test's stand-in where the row itself is under test.
type statRecorder interface {
	Record(routing.ProviderConfig, routing.Stat)
}

// Session is a live model attached to one customer. It hands out the provider's streams
// untouched apart from recording a stat row per response on the way past.
type Session struct {
	mu       sync.Mutex
	closed   bool
	children map[*Session]struct{}
	fallback func(context.Context, llm.ResponseParams) (*llm.Stream, error)
	provider Provider
	// config is the routing identity of the provider. Stats and health are keyed by it,
	// so a provider registered under a different name still aggregates coherently.
	config   routing.ProviderConfig
	owner    routing.Owner
	recorder statRecorder
	// quota caps what the owner's end user may spend in a day. Nil caps nothing.
	quota *quota.Limiter
	// admit asks the owner's policies before each response, and screen judges what each
	// response is asked. Both are nil on a fallback child, which serves a response its
	// parent already admitted and screened.
	admit  func(context.Context, string) (routing.Admission, error)
	screen Screen
}

func newSession(
	provider Provider,
	config routing.ProviderConfig,
	owner routing.Owner,
	recorder *routing.Recorder,
	limiter *quota.Limiter,
) *Session {
	return &Session{
		provider: provider,
		config:   config,
		owner:    owner,
		recorder: recorder,
		quota:    limiter,
	}
}

// Create asks the selected provider for a response. The stream it returns records what the
// response cost as it is drained, which is why a caller must drain it even after closing
// it: an abandoned response still generated tokens.
func (s *Session) create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	startedAt := time.Now().UTC()

	ctx, span := tracer.Start(ctx, "llm.provider.create",
		trace.WithAttributes(
			attribute.String("llm.provider", s.config.Provider),
			attribute.String("llm.model", s.config.Model)))
	stream, err := s.provider.Create(ctx, params)
	tracing.Fail(span, err)
	span.End()
	if err != nil {
		durationMs := float64(time.Since(startedAt).Microseconds()) / 1000
		code := createErrorCode(ctx)
		s.recorder.Record(s.config, routing.Stat{
			Owner:        s.owner,
			StartedAt:    startedAt,
			OperationID:  params.ID,
			Purpose:      params.Purpose,
			TurnID:       params.TurnID,
			DurationMs:   durationMs,
			Success:      false,
			ErrorCode:    code,
			ErrorMessage: err.Error(),
		})
		slog.Info("model call timing", "call", s.owner.CallID, "operation", params.ID,
			"purpose", params.Purpose, "turn", params.TurnID,
			"provider", s.config.Provider, "model", s.config.Model,
			"duration_ms", durationMs, "success", false)
		if params.OnTiming != nil {
			params.OnTiming(llm.CallTiming{OperationID: params.ID, Purpose: params.Purpose,
				TurnID: params.TurnID, Provider: s.config.Provider, Model: s.config.Model,
				DurationMs: durationMs, Success: false})
		}
		return nil, err
	}
	return stream.Observe(func(event llm.Event) { s.observe(startedAt, params, event) }), nil
}

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Capabilities is what the model serving this session accepts.
func (s *Session) Capabilities() llm.Capabilities { return s.provider.Capabilities() }

// Price is what this session's provider charges, so a caller can report a cost without
// reaching for the router's config.
func (s *Session) Price() routing.Price { return s.config.Price }

// ContextWindow is how many tokens of prompt this session's model takes, or zero when
// its config does not say.
func (s *Session) ContextWindow() int64 { return s.config.ContextWindow }

// LLM exposes the underlying provider so callers can reach provider-specific features.
func (s *Session) LLM() llm.LLM { return s.provider }

// Close ends the session, abandoning anything still in flight.
func (s *Session) Close() error {
	s.mu.Lock()
	s.closed = true
	children := make([]*Session, 0, len(s.children))
	for child := range s.children {
		children = append(children, child)
	}
	s.mu.Unlock()
	var failures []error
	for _, child := range children {
		failures = append(failures, child.Close())
	}
	return stack.Wrap(errors.Join(append(failures, s.provider.Close())...))
}

// observe records statistics as a response settles.
//
// One response is one unit of billable work, the way one synthesis is for text-to-speech,
// and it is recorded once: a failure is carried on the response that failed rather than
// written as a row of its own, so one turn stays one row.
func (s *Session) observe(startedAt time.Time, params llm.ResponseParams, event llm.Event) {
	completed, settled := event.(llm.ResponseCompleted)
	if !settled {
		return
	}

	response := completed.Response
	slog.Info("model call timing", "call", s.owner.CallID, "operation", response.ID,
		"purpose", params.Purpose, "turn", params.TurnID,
		"provider", s.config.Provider, "model", s.config.Model,
		"ttft_ms", response.TimeToFirstTokenMs, "duration_ms", response.DurationMs,
		"success", response.Status != llm.StatusFailed)
	if params.OnTiming != nil {
		params.OnTiming(llm.CallTiming{OperationID: response.ID, Purpose: params.Purpose,
			TurnID: params.TurnID, Provider: s.config.Provider, Model: s.config.Model,
			TTFTMs: response.TimeToFirstTokenMs, DurationMs: response.DurationMs,
			Success: response.Status != llm.StatusFailed})
	}

	// The debit is taken here rather than where the response was asked for, because what a
	// response costs is only known once it has settled. A caller who slipped in under the
	// limit and then spent the rest of the day's tokens in one answer is refused the next
	// one, which is the most a limit counted after the fact can do.
	//
	// Its context is not the one the response was created with: that one is cancelled when
	// the caller hangs up, and a response generated by somebody who hung up before reading
	// it cost exactly as much as one they read.
	s.quota.Debit(context.Background(), s.owner.CustomerID, s.owner.Caller,
		response.Usage.InputTokens+response.Usage.OutputTokens)

	composed := llm.Compose(params).Scaled(response.Usage.InputTokens)
	s.recorder.Record(s.config, routing.Stat{
		Owner:       s.owner,
		StartedAt:   startedAt,
		OperationID: response.ID,
		Purpose:     params.Purpose,
		TurnID:      params.TurnID,
		DurationMs:  response.DurationMs,
		Usage: routing.Usage{
			InputTokens:       response.Usage.InputTokens,
			CachedInputTokens: response.Usage.InputTokensDetails.CachedTokens,
			OutputTokens:      response.Usage.OutputTokens,
		},
		InputParts: store.InputParts{
			InstructionTokens:    composed.Instructions,
			MessageTokens:        composed.Messages,
			ToolDefinitionTokens: composed.ToolDefinitions,
			ToolUseTokens:        composed.ToolUse,
			ImageTokens:          composed.Images,
			VideoTokens:          composed.Video,
		},
		// Time to first token is what the caller actually waited for; the rest of the
		// answer arrives while they are already reading or hearing it.
		LatencyMs: measuredLatency(response, response.TimeToFirstTokenMs),
		Success:   served(response),
		ErrorCode: errorCode(response),
	})
}

// served reports whether a response ran to its end as asked. One the caller closed first
// did not, and neither did one that failed.
func served(response llm.Response) bool {
	return response.Status != llm.StatusFailed && response.Status != llm.StatusCancelled
}

// errorCode says why a response did not run to its end: the provider failing it, or its
// caller closing the stream first, which is no fault of the provider's. A response that
// ended as asked has none.
func errorCode(response llm.Response) string {
	switch response.Status {
	case llm.StatusFailed:
		return "provider_error"
	case llm.StatusCancelled:
		return routing.ErrorCancelled
	}
	return ""
}

// measuredLatency is the time the caller waited for the first token, or nothing when none
// ever came. A response that produced nothing reports no time to first token, and the wait
// for its headers alone is not one: counted as such, a stream closed the moment it opened
// would look like the fastest the provider ever answered.
func measuredLatency(response llm.Response, ttftMs float64) float64 {
	if response.TimeToFirstTokenMs <= 0 {
		return 0
	}
	return ttftMs
}

// Only requests that have not started streaming can be replayed safely. A partial
// answer or tool call is never retried, and provider-held history cannot transfer.
func (s *Session) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return nil, stack.Wrap(errors.New("llmrouter: session is closed"))
	}
	ctx, span := tracer.Start(ctx, "llm.create")
	defer span.End()
	// The limit is asked here rather than only when the session was opened, because a
	// session answers many turns: a socket goes on sending frames and a call goes on
	// talking long after whatever opened it was let through.
	if err := s.quota.Allow(ctx, s.owner.CustomerID, s.owner.Caller); err != nil {
		return nil, stack.Wrap(err)
	}
	if s.admit != nil {
		if _, err := s.admit(ctx, s.owner.CustomerID); err != nil {
			return nil, stack.Wrap(err)
		}
	}
	// The screen is started before the model is asked so the two overlap from the first
	// byte, rather than the screen waiting on however long the provider takes to accept.
	var verdict <-chan error
	if s.screen != nil {
		verdict = s.screen(ctx, s.owner, params.Input)
	}
	stream, err := s.create(ctx, params)
	if err == nil || ctx.Err() != nil || s.fallback == nil || params.PreviousResponseID != "" || params.Conversation != "" {
		return screened(stream, verdict), stack.Wrap(err)
	}
	var apiError *openai.Error
	// 402 is the provider's billing, not the request: Gemini answers it for every request
	// once prepaid credit runs out, and a call pinned to it would never be answered again.
	if errors.As(err, &apiError) && apiError.StatusCode < 500 && apiError.StatusCode != 401 && apiError.StatusCode != 402 && apiError.StatusCode != 403 && apiError.StatusCode != 404 && apiError.StatusCode != 408 && apiError.StatusCode != 429 {
		return nil, stack.Wrap(err)
	}
	alternate, fallbackErr := s.fallback(ctx, params)
	if fallbackErr != nil {
		return nil, stack.Wrap(errors.Join(err, fallbackErr))
	}
	return screened(alternate, verdict), nil
}

// screened attaches a verdict to a stream, when there is both.
func screened(stream *llm.Stream, verdict <-chan error) *llm.Stream {
	if stream == nil || verdict == nil {
		return stream
	}
	return stream.Screen(verdict)
}

func (s *Session) addChild(child *Session) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	if s.children == nil {
		s.children = map[*Session]struct{}{}
	}
	s.children[child] = struct{}{}
	return true
}
func (s *Session) releaseChild(child *Session) {
	_ = child.Close()
	s.mu.Lock()
	delete(s.children, child)
	s.mu.Unlock()
}

// createErrorCode says why a request ended before the provider answered it: the caller
// cancelling it, which is no fault of the provider's, or the provider failing it.
func createErrorCode(ctx context.Context) string {
	if ctx.Err() != nil {
		return routing.ErrorCancelled
	}
	return "create_failed"
}
