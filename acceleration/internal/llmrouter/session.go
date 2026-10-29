package llmrouter

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/mock"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
)

var tracer = tracing.Tracer("llmrouter")

// statRecorder is where a session writes the stat row of each response, and what prices
// it. It is the router's recorder, and a test's stand-in where the row itself is under
// test.
type statRecorder interface {
	Record(routing.ProviderConfig, routing.Stat)
	QuoteStat(routing.ProviderConfig, routing.Stat) pricing.Quote
	Billable(routing.ProviderConfig) bool
	Reserve(routing.ProviderConfig, routing.Owner, uuid.UUID, pricing.Quantities, time.Duration) error
	Release(uuid.UUID)
}

// holdOutputTokens is what a response that sets no cap is reserved for generating.
const holdOutputTokens = 4_096

// reserveRenewEvery is how often the reservation of a response still streaming is renewed, for
// twice as long, so a long answer stays held until its row settles it; maxReserved is how long
// one is renewed at most, for a stream nobody drains.
var (
	reserveRenewEvery = routing.ReserveTTL / 2
	maxReserved       = time.Hour
)

// reservation is the money set aside for one response before it was asked for, under the
// usage id its row carries, and the estimate it was set aside for. Only the first attempt to
// record takes the id: a hedge or a fallback asked for the same response records under one of
// its own.
type reservation struct {
	id       uuid.UUID
	estimate pricing.Quantities
	claimed  atomic.Bool
}

type reservationKey struct{}

// claim is the reservation for the response ctx asks for, the first time it is asked; nil after
// that, or when nothing was reserved.
func claim(ctx context.Context) *reservation {
	held, _ := ctx.Value(reservationKey{}).(*reservation)
	if held == nil || !held.claimed.CompareAndSwap(false, true) {
		return nil
	}
	return held
}

// idOf is the usage id reserved under, nil for none.
func (r *reservation) idOf() uuid.UUID {
	if r == nil {
		return uuid.Nil
	}
	return r.id
}

// keep renews the reservation while its response streams, until the returned func is called
// (it settled) or maxReserved has passed. A renewal refused keeps what was held until it
// expires.
func (s *Session) keep(held *reservation) func() {
	if held == nil {
		return func() {}
	}
	done := make(chan struct{})
	var once sync.Once
	go func() {
		ticker := time.NewTicker(reserveRenewEvery)
		defer ticker.Stop()
		limit := time.NewTimer(maxReserved)
		defer limit.Stop()
		for {
			select {
			case <-done:
				return
			case <-s.closing:
				return
			case <-limit.C:
				return
			case <-ticker.C:
				_ = s.recorder.Reserve(s.config, s.owner, held.id, held.estimate, 2*reserveRenewEvery)
			}
		}
	}()
	return func() { once.Do(func() { close(done) }) }
}

// reserve sets aside what the response may cost, on its prompt and on the most it may
// generate, before the provider is asked; ctx carries what was reserved to the row.
func (s *Session) reserve(ctx context.Context, params llm.ResponseParams) (context.Context, error) {
	output := int64(params.MaxOutputTokens)
	if output <= 0 {
		output = holdOutputTokens
	}
	estimate := pricing.Quantities{pricing.InputTokens: llm.Compose(params).Total(), pricing.OutputTokens: output}
	id := uuid.Must(uuid.NewV7())
	if err := s.recorder.Reserve(s.config, s.owner, id, estimate, 2*reserveRenewEvery); err != nil {
		return ctx, err
	}
	return context.WithValue(ctx, reservationKey{}, &reservation{id: id, estimate: estimate}), nil
}

// errClosed is what a session answers once it has been closed.
var errClosed = errors.New("llmrouter: session is closed")

// Session is a live model attached to one customer. It hands out the provider's streams
// untouched apart from recording a stat row per response on the way past.
type Session struct {
	mu     sync.Mutex
	closed bool
	// closing closes with the session, which stops renewing its responses' reservations.
	closing   chan struct{}
	closeOnce sync.Once
	children  map[*Session]struct{}
	fallback  func(context.Context, llm.ResponseParams) (*llm.Stream, error)
	// hedge opens a session on another candidate of the same target, and hedgeAfter is how
	// long a reply may say nothing before it is asked of that one as well, flowHedgeAfter the
	// same for a flow decision. All are unset on a fallback child, which is only ever asked for
	// a response its parent has chosen it for.
	hedge          func(context.Context) (*Session, error)
	hedgeAfter     time.Duration
	flowHedgeAfter time.Duration
	provider       Provider
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
	admit  func(context.Context, routing.Owner) (routing.Admission, error)
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
		closing:  make(chan struct{}),
	}
}

// Create asks the selected provider for a response. The stream it returns records what the
// response cost as it is drained, which is why a caller must drain it even after closing
// it: an abandoned response still generated tokens.
func (s *Session) create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	return s.attempt(ctx, params, nil)
}

// attempt is create for one attempt of a hedged request, which its row names.
func (s *Session) attempt(ctx context.Context, params llm.ResponseParams, hedge *hedgeAttempt) (*llm.Stream, error) {
	startedAt := time.Now().UTC()

	ctx, span := tracer.Start(ctx, "llm.provider.create",
		trace.WithAttributes(
			attribute.String("llm.provider", s.config.Provider),
			attribute.String("llm.model", s.config.Model)))
	stream, err := s.provider.Create(ctx, params)
	tracing.Fail(span, err)
	span.End()
	held := claim(ctx)
	reserved := held.idOf()
	owner := s.owner
	// A response asked for with a Spent of its own, as the LLM socket asks each one, adds its
	// cost up apart from the session's.
	if spent := routing.SpentFrom(ctx); spent != nil {
		owner.Spent = spent
	}
	if err != nil {
		// The provider took nothing on, so nothing it would bill is held.
		s.recorder.Release(reserved)
		durationMs := float64(time.Since(startedAt).Microseconds()) / 1000
		code := outrun(createErrorCode(ctx), true, hedge)
		message := err.Error()
		if s.config.BYOK && code != routing.ErrorCancelled && keyRefused(err) {
			code = routing.ErrorKeyRefused
			message = fmt.Sprintf("%s refused the key this app keeps for it in %s, so the deployment's was asked next: %s",
				s.config.Provider, cmp.Or(s.config.Region, pricing.DefaultRegion), message)
		}
		stat := routing.Stat{
			Owner:        owner,
			StartedAt:    startedAt,
			OperationID:  params.ID,
			Purpose:      params.Purpose,
			TurnID:       params.TurnID,
			DurationMs:   durationMs,
			Success:      false,
			ErrorCode:    code,
			ErrorMessage: message,
		}
		label(&stat, params, hedge)
		s.recorder.Record(s.config, stat)
		slog.Info("model call timing", "call", s.owner.CallID, "operation", params.ID,
			"purpose", params.Purpose, "turn", params.TurnID,
			"provider", s.config.Provider, "model", s.config.Model,
			"duration_ms", durationMs, "success", false, "error_code", code)
		if params.OnTiming != nil {
			params.OnTiming(llm.CallTiming{OperationID: params.ID, Purpose: params.Purpose,
				TurnID: params.TurnID, Provider: s.config.Provider, Model: s.config.Model,
				DurationMs: durationMs, Success: false, ErrorCode: code})
		}
		return nil, err
	}
	createdAt := time.Now()
	charged := s.charged(ctx)
	settled := s.keep(held)
	return stream.Observe(func(event llm.Event) {
		if _, done := event.(llm.ResponseCompleted); done {
			settled()
		}
		s.observe(startedAt, createdAt, params, hedge, event, owner, charged, reserved)
	}), nil
}

// label puts on a response's row what its caller and its hedge say about it. The response
// is kept only if nothing that could throw it away did: a speculative reply that was dropped
// is not kept however its hedge ended, and neither is a hedge's losing attempt.
func label(stat *routing.Stat, params llm.ResponseParams, hedge *hedgeAttempt) {
	if params.Attributes != nil {
		for name, value := range params.Attributes() {
			stat.Attribute(name, value)
		}
	}
	id, kept, hedged := hedge.outcome()
	if !hedged {
		return
	}
	stat.Attribute(routing.AttributeHedge, id)
	kept = kept && stat.Attributes[routing.AttributeKept] != "false"
	stat.Attribute(routing.AttributeKept, strconv.FormatBool(kept))
}

// Provider is the provider serving this session.
func (s *Session) Provider() string { return s.config.Provider }

// Model is the model serving this session.
func (s *Session) Model() string { return s.config.Model }

// Capabilities is what the model serving this session accepts.
func (s *Session) Capabilities() llm.Capabilities { return s.provider.Capabilities() }

// Quote is what that usage of this session's provider costs on its tokens alone, with no
// cache writes and at the default service tier. QuoteResponse prices a response as its
// row is.
func (s *Session) Quote(usage routing.Usage) pricing.Quote {
	return s.recorder.QuoteStat(s.config, routing.Stat{Usage: usage})
}

// QuoteResponse is what a response from this session's provider costs, priced as its row
// is, so a caller can report a cost without reaching for the router's prices.
func (s *Session) QuoteResponse(response llm.Response) pricing.Quote {
	return s.recorder.QuoteStat(s.config, reported(response))
}

// ContextWindow is how many tokens of prompt this session's model takes, or zero when
// its config does not say.
func (s *Session) ContextWindow() int64 { return s.config.ContextWindow }

// Wrote reports whether the session's own provider wrote the response, rather than a hedge or
// a failover behind the session. A response is stamped with the names its adapter gives, which
// need not be the ones the session is routed under.
func (s *Session) Wrote(response llm.Response) bool {
	return response.Provider == s.provider.Provider() && response.Model == s.provider.Model()
}

// Close ends the session, abandoning anything still in flight.
func (s *Session) Close() error {
	// No reservation is renewed past the session: a response still settling is held until
	// its last renewal expires.
	s.closeOnce.Do(func() {
		if s.closing != nil {
			close(s.closing)
		}
	})
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
func (s *Session) observe(startedAt, createdAt time.Time, params llm.ResponseParams, hedge *hedgeAttempt, event llm.Event, owner routing.Owner, charged routing.Caller, reserved uuid.UUID) {
	completed, settled := event.(llm.ResponseCompleted)
	if !settled {
		return
	}

	response := completed.Response
	// A provider may wait for response headers before returning its stream. The stream's
	// own clock starts only then, but the caller has been waiting since Create began.
	headersMs := float64(createdAt.Sub(startedAt).Microseconds()) / 1000
	ttftMs := measuredLatency(response, headersMs+response.TimeToFirstTokenMs)
	durationMs := headersMs + response.DurationMs
	code := outrun(errorCode(response), response.TimeToFirstTokenMs <= 0, hedge)
	stat := reported(response)
	estimated := unreported(response)
	if estimated {
		// The provider never said what the response used: billed on the router's own count
		// and marked, so it is neither free nor silent.
		stat = routing.Stat{Attributes: stat.Attributes}
		stat.Use(llm.Estimate(params, response))
		stat.Estimated()
	}
	slog.Info("model call timing", "call", s.owner.CallID, "operation", response.ID,
		"purpose", params.Purpose, "turn", params.TurnID,
		"provider", s.config.Provider, "model", s.config.Model,
		"ttft_ms", ttftMs, "duration_ms", durationMs,
		"input_tokens", stat.InputTokens,
		"cached_tokens", stat.CachedInputTokens,
		"cache_write_tokens", response.Usage.InputTokensDetails.CacheWriteTokens,
		"output_tokens", stat.OutputTokens,
		"reasoning_tokens", response.Usage.OutputTokensDetails.ReasoningTokens,
		"estimated", estimated,
		"success", served(response), "error_code", code)
	composed := llm.Compose(params).Scaled(stat.InputTokens)
	usage := stat.Usage
	stat.Owner = owner
	// Recorded under the id it was reserved under, which settles the reservation at the cost.
	stat.UsageID = reserved
	stat.StartedAt = startedAt
	stat.OperationID = response.ID
	stat.Purpose = params.Purpose
	stat.TurnID = params.TurnID
	stat.DurationMs = durationMs
	stat.InputParts = store.InputParts{
		InstructionTokens:    composed.Instructions,
		MessageTokens:        composed.Messages,
		ToolDefinitionTokens: composed.ToolDefinitions,
		ToolUseTokens:        composed.ToolUse,
		ImageTokens:          composed.Images,
		VideoTokens:          composed.Video,
	}
	// Time to first token is what the caller actually waited for; the rest of the answer
	// arrives while they are already reading or hearing it.
	stat.LatencyMs = measuredLatency(response, ttftMs)
	stat.Success = served(response)
	stat.ErrorCode = code
	label(&stat, params, hedge)
	if params.OnTiming != nil {
		params.OnTiming(llm.CallTiming{OperationID: response.ID, Purpose: params.Purpose,
			TurnID: params.TurnID, Provider: s.config.Provider, Model: s.config.Model,
			TTFTMs: ttftMs, DurationMs: durationMs,
			Success: served(response), ErrorCode: code,
			InputTokens: usage.InputTokens, CachedInputTokens: usage.CachedInputTokens,
			OutputTokens: usage.OutputTokens, CostMicros: s.recorder.QuoteStat(s.config, stat).CostMicros})
	}

	// The debit is taken here rather than where the response was asked for, because what a
	// response costs is only known once it has settled. A caller who slipped in under the
	// limit and then spent the rest of the day's tokens in one answer is refused the next
	// one, which is the most a limit counted after the fact can do.
	//
	// Its context is not the one the response was created with: that one is cancelled when
	// the caller hangs up, and a response generated by somebody who hung up before reading
	// it cost exactly as much as one they read.
	s.debit(charged, stat.InputTokens+stat.OutputTokens)

	s.recorder.Record(s.config, stat)
}

// reported is what a response's row is priced by: its tokens, the provider's own total to
// check them against, and what the provider said beyond them.
func reported(response llm.Response) routing.Stat {
	stat := routing.Stat{VendorTotalTokens: response.Usage.TotalTokens, Attributes: reportedAttributes(response)}
	stat.Use(response.Used())
	stat.VendorCost(response)
	return stat
}

// unreported reports whether a response's row has to be estimated: its provider never said
// what it used, and it either produced something or was cut short, by its caller or by a
// deadline, which most providers then never report but bill all the same, the whole prompt
// included. A provider that repeats its usage on every chunk has said, and a response cut
// short keeps that count. A response the provider itself failed is not estimated: a vendor is
// not taken to bill its own failure.
func unreported(response llm.Response) bool {
	if response.UsageReported || (response.Status == llm.StatusFailed && !response.TimedOut) {
		return false
	}
	produced := response.OutputText != "" || response.ReasoningText != "" || len(response.ToolCalls) > 0
	return produced || response.Status == llm.StatusCancelled || response.TimedOut
}

// reportedAttributes is what the provider said about how it served the response: the tier,
// which chooses the price, and the reasoning, which is shown and never priced, since the
// output tokens already include it.
func reportedAttributes(response llm.Response) map[string]string {
	attributes := map[string]string{}
	if response.ServiceTier != "" {
		attributes["service_tier"] = response.ServiceTier
	}
	if reasoning := response.Usage.OutputTokensDetails.ReasoningTokens; reasoning != 0 {
		attributes["reasoning_tokens"] = strconv.FormatInt(reasoning, 10)
	}
	if len(attributes) == 0 {
		return nil
	}
	return attributes
}

// served reports whether a response ran to its end as asked. One the caller closed first
// did not, and neither did one that failed.
func served(response llm.Response) bool {
	return response.Status != llm.StatusFailed && response.Status != llm.StatusCancelled
}

// errorCode says why a response did not run to its end: the provider failing it, a deadline
// running out, or its caller closing the stream first, which is no fault of the provider's. A
// response that ended as asked has none.
func errorCode(response llm.Response) string {
	if response.TimedOut {
		return routing.ErrorTimeout
	}
	switch response.Status {
	case llm.StatusFailed:
		return "provider_error"
	case llm.StatusCancelled:
		return routing.ErrorCancelled
	}
	return ""
}

// outrun is code, unless the attempt was cancelled because it lost its hedge without having
// said anything: then it ran past the router's own limit on silence, a timeout, which counts
// against its provider's health where a cancellation would not.
func outrun(code string, silent bool, hedge *hedgeAttempt) string {
	if code == routing.ErrorCancelled && silent && hedge.stalled() {
		return routing.ErrorTimeout
	}
	return code
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
		return nil, stack.Wrap(errClosed)
	}
	ctx, span := tracer.Start(ctx, "llm.create")
	defer span.End()
	// The limit is asked here rather than only when the session was opened, because a
	// session answers many turns: a socket goes on sending frames and a call goes on
	// talking long after whatever opened it was let through.
	if err := s.allow(ctx); err != nil {
		return nil, err
	}
	if s.admit != nil {
		admission, err := s.admit(ctx, s.owner)
		if err != nil {
			return nil, stack.Wrap(err)
		}
		// Money runs out mid-session too. A model that costs nothing keeps answering, and a
		// fallback or a hedge is admitted again on its own model.
		if admission.Spend != nil && s.recorder.Billable(s.config) {
			return nil, stack.Wrap(admission.Spend)
		}
		// Admitted, the response's money is set aside before the provider is asked, so the
		// other routers count it while it is generated.
		if ctx, err = s.reserve(ctx, params); err != nil {
			return nil, stack.Wrap(err)
		}
	}
	// The screen is started before the model is asked so the two overlap from the first
	// byte, rather than the screen waiting on however long the provider takes to accept.
	var verdict <-chan error
	if s.screen != nil {
		verdict = s.screen(ctx, s.owner, params.Input)
	}
	open := s.create
	if s.hedges(params) {
		open = s.hedged
	}
	stream, err := open(ctx, params)
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

// counted reports whether this session's answers count toward a daily limit. A mock
// answer costs nothing and is what a load test asks for by the thousand.
func (s *Session) counted() bool { return s.config.Provider != mock.ProviderName }

// charged is whose day the turn ctx asks for is counted on. In a session a backend opened,
// it is whoever sent the turn: the backend, which is never counted, or the user the session
// was opened for, with their own token. Any other session's turns are its opener's, as a
// session only its opener and their backend can reach.
func (s *Session) charged(ctx context.Context) routing.Caller {
	// A turn a scheduled task fired is the customer's spend, never the end
	// user's: they agreed to the task once, not to each time it runs.
	if routing.ScheduledTurn(ctx) {
		return routing.Caller{ServerSide: true}
	}
	if sender, ok := routing.CallerFrom(ctx); ok && s.owner.Caller.ServerSide {
		return sender
	}
	return s.owner.Caller
}

// allow asks the daily limit whether this session may answer the turn ctx asks for. A
// session that is not counted is never asked.
func (s *Session) allow(ctx context.Context) error {
	if !s.counted() {
		return nil
	}
	if _, err := s.quota.Allow(ctx, s.owner.CustomerID, s.charged(ctx)); err != nil {
		return stack.Wrap(err)
	}
	return nil
}

// debit charges the daily limit for the tokens a settled response used. A session that is
// not counted is never charged.
func (s *Session) debit(charged routing.Caller, tokens int64) {
	if !s.counted() {
		return
	}
	s.quota.Debit(context.Background(), s.owner.CustomerID, charged, tokens)
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

// keyRefused reports whether the vendor refused the key a request was made with: 401 for a key
// it does not know, 403 for one it does not let at the model. Every vendor whose factory takes a
// key of the instance's own answers through OpenAI's client or Anthropic's.
func keyRefused(err error) bool {
	status := 0
	if refused, ok := errors.AsType[*openai.Error](err); ok {
		status = refused.StatusCode
	} else if refused, ok := errors.AsType[*anthropic.Error](err); ok {
		status = refused.StatusCode
	}
	return status == http.StatusUnauthorized || status == http.StatusForbidden
}

// createErrorCode says why a request ended before the provider answered it: the caller
// cancelling it, which is no fault of the provider's, or the provider failing it.
func createErrorCode(ctx context.Context) string {
	if ctx.Err() != nil {
		return routing.ErrorCancelled
	}
	return "create_failed"
}
