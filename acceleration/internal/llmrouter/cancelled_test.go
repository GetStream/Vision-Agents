package llmrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestARequestItsCallerCancelledIsNotAProviderFailure(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	if got := createErrorCode(ctx); got != "create_failed" {
		t.Fatalf("a request that failed on its own was recorded as %q", got)
	}
	cancel()
	if got := createErrorCode(ctx); got != routing.ErrorCancelled {
		t.Fatalf("a request its caller cancelled was recorded as %q", got)
	}
}

// statSink keeps the rows a session writes, where a recorder with no store would drop them.
type statSink struct {
	mu   sync.Mutex
	rows []routing.Stat
}

func (k *statSink) Record(_ routing.ProviderConfig, row routing.Stat) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.rows = append(k.rows, row)
}

// Quote prices nothing: the rows, not what they cost, are under test.
func (k *statSink) QuoteStat(routing.ProviderConfig, routing.Stat) pricing.Quote {
	return pricing.Quote{}
}

// only is the single row the session wrote.
func (k *statSink) only(t *testing.T) routing.Stat {
	t.Helper()
	k.mu.Lock()
	defer k.mu.Unlock()
	require.Len(t, k.rows, 1, "one response is one row")
	return k.rows[0]
}

// headerWait is how long the stub provider takes to answer with its headers, which is the
// only time a stream closed before it says anything has to its name.
const headerWait = 20 * time.Millisecond

// sinkSession is a session over a stub provider that answers its headers after headerWait.
func sinkSession(t *testing.T) (*Session, *stubLLM, *statSink) {
	t.Helper()
	provider := newStubLLM()
	provider.createDelay = headerWait
	session := newSession(provider, routing.ProviderConfig{Provider: "stub", Model: "stub-model"},
		routing.Owner{CustomerID: "acme"}, nil, nil)
	sink := &statSink{}
	session.recorder = sink
	t.Cleanup(func() { _ = session.Close() })
	return session, provider, sink
}

func TestAStreamClosedBeforeItSaidAnythingIsNeitherASuccessNorAFastOne(t *testing.T) {
	session, _, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	require.NoError(t, stream.Close())
	require.Equal(t, llm.StatusCancelled, drain(stream).Status)

	row := sink.only(t)
	require.False(t, row.Success, "a response its caller closed was not served")
	require.Equal(t, routing.ErrorCancelled, row.ErrorCode)
	require.Zero(t, row.LatencyMs, "the wait for the headers alone is not a time to first token")
	require.Greater(t, row.DurationMs, 0.0, "the request is still logged with how long it ran")
}

func TestAStreamClosedAfterItsFirstTokenIsStillBilledForWhatItGenerated(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 4})
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	require.Equal(t, llm.StatusCancelled, drain(stream).Status)

	row := sink.only(t)
	require.False(t, row.Success)
	require.Equal(t, routing.ErrorCancelled, row.ErrorCode)
	require.EqualValues(t, 10, row.Usage.InputTokens, "the prompt was paid for")
	require.EqualValues(t, 4, row.Usage.OutputTokens, "and so was what the model said before it was cut off")
	require.Greater(t, row.LatencyMs, 0.0,
		"the first token did arrive, so what the caller waited for it is known")
}

func TestAStreamCutBeforeItsUsageArrivedIsBilledOnAnEstimate(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	require.Equal(t, llm.StatusCancelled, drain(stream).Status)

	row := sink.only(t)
	require.Equal(t, routing.ErrorCancelled, row.ErrorCode)
	require.EqualValues(t, 1, row.Usage.InputTokens, "the prompt the vendor read")
	require.EqualValues(t, 3, row.Usage.OutputTokens, "and what it said before the cut")
	require.Equal(t, map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
}

func TestTheTimingLogShowsWhatAnEstimatedResponseIsBilled(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	session, provider, _ := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	drain(stream)

	logged := timingLogged(t, &logs)
	require.True(t, logged.Estimated)
	require.EqualValues(t, 1, logged.InputTokens, "what the row bills, not the nothing the vendor said")
	require.EqualValues(t, 3, logged.OutputTokens)
}

func TestAStreamCutAfterItsUsageArrivedKeepsTheVendorsCount(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 4})
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	drain(stream)

	row := sink.only(t)
	require.EqualValues(t, 10, row.Usage.InputTokens)
	require.EqualValues(t, 4, row.Usage.OutputTokens)
	require.Empty(t, row.Attributes, "a vendor that repeats its usage on every chunk has said")
}

func TestAFailedStreamIsNotEstimated(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Fail(errors.New("upstream reset"), "stream")
	drain(stream)

	row := sink.only(t)
	require.Zero(t, row.Usage.InputTokens)
	require.Empty(t, row.Attributes)
}

func TestAnAnswerThatSaidNothingAndReportedNothingIsLeftForTheAlarm(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Done()
	drain(stream)

	row := sink.only(t)
	require.Zero(t, row.Usage.InputTokens, "nothing came back to estimate from")
	require.Empty(t, row.Attributes)
}

func TestAStreamThatRanToItsEndIsASuccessWithItsLatency(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("Hello")
	provider.script(0).Done()
	require.Equal(t, llm.StatusCompleted, drain(stream).Status)

	row := sink.only(t)
	require.True(t, row.Success)
	require.Empty(t, row.ErrorCode)
	require.Greater(t, row.LatencyMs, 0.0)
}

func TestAStreamThatEndedWithoutSayingAnythingHasNoLatency(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Done()
	require.Equal(t, llm.StatusCompleted, drain(stream).Status)

	row := sink.only(t)
	require.True(t, row.Success, "it ended as the provider meant it to")
	require.Zero(t, row.LatencyMs, "nothing was said, so there was no first token to wait for")
}

func TestAFailedStreamIsAProviderErrorWhetherOrNotItSaidAnything(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Fail(errors.New("upstream reset"), "stream")
	require.Equal(t, llm.StatusFailed, drain(stream).Status)

	row := sink.only(t)
	require.False(t, row.Success)
	require.Equal(t, "provider_error", row.ErrorCode)
}

func TestWhatEndsAResponseDecidesHowItIsCounted(t *testing.T) {
	for _, test := range []struct {
		status    llm.ResponseStatus
		served    bool
		errorCode string
	}{
		{llm.StatusCompleted, true, ""},
		{llm.StatusIncomplete, true, ""},
		{llm.StatusFailed, false, "provider_error"},
		{llm.StatusCancelled, false, routing.ErrorCancelled},
	} {
		response := llm.Response{Status: test.status}
		require.Equal(t, test.served, served(response), string(test.status))
		require.Equal(t, test.errorCode, errorCode(response), string(test.status))
	}
}

func TestOnlyAFirstTokenMakesALatency(t *testing.T) {
	require.Zero(t, measuredLatency(llm.Response{}, 20), "headers alone are no first token")
	require.Equal(t, 35.0, measuredLatency(llm.Response{TimeToFirstTokenMs: 15}, 35))
}

// loggedTiming is a line the session logged, read back from JSON.
type loggedTiming struct {
	Message      string `json:"msg"`
	Success      bool   `json:"success"`
	ErrorCode    string `json:"error_code"`
	InputTokens  int64  `json:"input_tokens"`
	OutputTokens int64  `json:"output_tokens"`
	Estimated    bool   `json:"estimated"`
}

// timingLogged is the one "model call timing" line among those logged.
func timingLogged(t *testing.T, logs *bytes.Buffer) loggedTiming {
	t.Helper()
	var found []loggedTiming
	for decoder := json.NewDecoder(logs); decoder.More(); {
		var line loggedTiming
		require.NoError(t, decoder.Decode(&line))
		if line.Message == "model call timing" {
			found = append(found, line)
		}
	}
	require.Len(t, found, 1)
	return found[0]
}

func TestAResponseIsTimedAsItEnded(t *testing.T) {
	for name, test := range map[string]struct {
		end       func(*testing.T, *llmtest.Script, *llm.Stream)
		success   bool
		errorCode string
	}{
		"run to its end": {func(_ *testing.T, script *llmtest.Script, _ *llm.Stream) { script.Done() }, true, ""},
		// A superseded flow decision, a losing hedge and a barge-in all end this way.
		"closed by its caller": {func(t *testing.T, _ *llmtest.Script, stream *llm.Stream) {
			require.NoError(t, stream.Close())
		}, false, routing.ErrorCancelled},
		"failed by the provider": {func(_ *testing.T, script *llmtest.Script, _ *llm.Stream) {
			script.Fail(errors.New("upstream reset"), "stream")
		}, false, "provider_error"},
	} {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })
			session, provider, sink := sinkSession(t)
			var timings []llm.CallTiming
			stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt(),
				OnTiming: func(timing llm.CallTiming) { timings = append(timings, timing) }})
			require.NoError(t, err)
			provider.script(0).OutputText("half a sen")
			provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 4})
			require.True(t, stream.Next())
			require.True(t, stream.Next())
			test.end(t, provider.script(0), stream)
			_ = drain(stream)

			require.Len(t, timings, 1)
			require.Equal(t, test.success, timings[0].Success)
			require.Equal(t, test.errorCode, timings[0].ErrorCode)
			row := sink.only(t)
			require.Equal(t, row.Success, timings[0].Success, "the call is reported live as it is stored")
			require.Equal(t, row.ErrorCode, timings[0].ErrorCode)
			require.EqualValues(t, 10, timings[0].InputTokens, "what it read is reported however it ended")
			require.EqualValues(t, 4, timings[0].OutputTokens)

			logged := timingLogged(t, &logs)
			require.Equal(t, test.success, logged.Success)
			require.Equal(t, test.errorCode, logged.ErrorCode)
		})
	}
}

func TestARequestThatWasNeverAnsweredIsTimedAsItEnded(t *testing.T) {
	for name, test := range map[string]struct {
		trouble   func(*rig, context.CancelFunc)
		errorCode string
	}{
		"refused by the provider": {func(r *rig, _ context.CancelFunc) { r.fail("alpha", errors.New("alpha is down")) }, "create_failed"},
		"given up by its caller": {func(r *rig, hangUp context.CancelFunc) {
			r.slow("alpha", time.Second)
			time.AfterFunc(time.Millisecond, hangUp)
		}, routing.ErrorCancelled},
	} {
		t.Run(name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := slog.Default()
			slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
			t.Cleanup(func() { slog.SetDefault(previous) })
			synctest.Test(t, func(t *testing.T) {
				r := newRig(t, 0, "alpha")
				ctx, hangUp := context.WithCancel(context.Background())
				defer hangUp()
				test.trouble(r, hangUp)

				_, err := r.session.Create(ctx, r.params())
				require.Error(t, err)

				require.Equal(t, map[string]string{"alpha": test.errorCode}, r.called())
			})
			logged := timingLogged(t, &logs)
			require.False(t, logged.Success)
			require.Equal(t, test.errorCode, logged.ErrorCode)
		})
	}
}
