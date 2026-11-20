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

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
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
	// cost is what every row is quoted at.
	cost int64
}

func (k *statSink) Record(_ routing.ProviderConfig, row routing.Stat) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.rows = append(k.rows, row)
}

// Quote prices every row at cost, nothing unless a test sets it: the rows, not what they
// cost, are under test.
func (k *statSink) QuoteStat(routing.ProviderConfig, routing.Stat) pricing.Quote {
	return pricing.Quote{CostMicros: k.cost}
}

// Billable is true: nothing here asks money of an admission.
func (k *statSink) Billable(routing.ProviderConfig) bool { return true }

func (k *statSink) Reserve(routing.ProviderConfig, routing.Owner, uuid.UUID, pricing.Quantities, time.Duration) error {
	return nil
}

func (k *statSink) Release(uuid.UUID) {}

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

func TestAStreamThatFailedAfterItsOutputIsBilledOnAnEstimate(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Fail(errors.New("upstream reset"), "stream")
	drain(stream)

	row := sink.only(t)
	require.Equal(t, "provider_error", row.ErrorCode)
	require.EqualValues(t, 1, row.Usage.InputTokens, "the vendor read the prompt before it failed")
	require.EqualValues(t, 3, row.Usage.OutputTokens, "and generated what streamed")
	require.Equal(t, map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
}

func TestAStreamThatFailedBeforeAnyOutputIsNotEstimated(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Fail(errors.New("upstream reset"), "stream")
	drain(stream)

	row := sink.only(t)
	require.Zero(t, row.Usage.InputTokens)
	require.Empty(t, row.Attributes)
}

func TestAnOutputCountTakenBeforeTheEndIsRaisedToWhatStreamed(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 1, OutputUnfinished: true})
	provider.script(0).OutputText("half a sen")
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	drain(stream)

	row := sink.only(t)
	require.EqualValues(t, 10, row.Usage.InputTokens, "the vendor's prompt count stands")
	require.EqualValues(t, 3, row.Usage.OutputTokens, "its opening output count did not")
	require.Equal(t, map[string]string{routing.AttributeUsageEstimated: "true"}, row.Attributes)
}

func TestAnUnfinishedOutputCountAboveWhatStreamedStands(t *testing.T) {
	session, provider, sink := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("half a sen")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 9, OutputUnfinished: true})
	require.True(t, stream.Next())
	require.True(t, stream.Next())
	require.NoError(t, stream.Close())
	drain(stream)

	row := sink.only(t)
	require.EqualValues(t, 9, row.Usage.OutputTokens)
	require.Empty(t, row.Attributes)
}

func TestAClosedStreamNobodyDrainedStillWritesItsRow(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		session, provider, sink := sinkSession(t)

		stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
		require.NoError(t, err)
		provider.script(0).OutputText("half a sen")
		require.True(t, stream.Next())
		require.True(t, stream.Next())
		require.NoError(t, stream.Close())

		time.Sleep(10 * time.Second)
		synctest.Wait()
		row := sink.only(t)
		require.Equal(t, routing.ErrorCancelled, row.ErrorCode)
		require.EqualValues(t, 3, row.Usage.OutputTokens, "what it said before it was walked away from")
	})
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
	Message            string  `json:"msg"`
	Success            bool    `json:"success"`
	ErrorCode          string  `json:"error_code"`
	InputTokens        int64   `json:"input_tokens"`
	OutputTokens       int64   `json:"output_tokens"`
	Estimated          bool    `json:"estimated"`
	AppID              string  `json:"app_id"`
	AgentID            *string `json:"agent_id"`
	CostMicros         int64   `json:"cost_micros"`
	Status             string  `json:"status"`
	IncompleteReason   string  `json:"incomplete_reason"`
	ProviderResponseID string  `json:"provider_response_id"`
	Error              string  `json:"error"`
	Transient          *bool   `json:"transient"`
	Region             string  `json:"region"`
	BYOK               bool    `json:"byok"`
}

// logTimings sends the default logger's lines to the buffer it returns, as JSON.
func logTimings(t *testing.T) *bytes.Buffer {
	t.Helper()
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	t.Cleanup(func() { slog.SetDefault(previous) })
	return &logs
}

func TestTheTimingLogNamesTheAppAgentAndWhatTheResponseCost(t *testing.T) {
	logs := logTimings(t)
	provider := newStubLLM()
	session := newSession(provider,
		routing.ProviderConfig{Provider: "stub", Model: "stub-model", Region: "eu-west", BYOK: true},
		routing.Owner{CustomerID: "acme", AgentID: "support", CallID: "c1"}, nil, nil)
	session.recorder = &statSink{cost: 1234}
	t.Cleanup(func() { _ = session.Close() })

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).ProviderResponseID("chatcmpl-9")
	provider.script(0).OutputText("hello")
	provider.script(0).Usage(llm.Usage{InputTokens: 10, OutputTokens: 2})
	provider.script(0).Done()
	drain(stream)

	logged := timingLogged(t, logs)
	require.Equal(t, "acme", logged.AppID, "the app, under the field the request lines use")
	require.NotNil(t, logged.AgentID)
	require.Equal(t, "support", *logged.AgentID)
	require.EqualValues(t, 1234, logged.CostMicros, "what the row is quoted at")
	require.Equal(t, string(llm.StatusCompleted), logged.Status)
	require.Equal(t, "chatcmpl-9", logged.ProviderResponseID, "the id to give the vendor")
	require.Equal(t, "eu-west", logged.Region)
	require.True(t, logged.BYOK, "asked on the app's own key")
	require.Empty(t, logged.Error, "a served response failed at nothing")
	require.Nil(t, logged.Transient)
}

func TestTheTimingLogNamesNoAgentOutsideAConversation(t *testing.T) {
	logs := logTimings(t)
	session, provider, _ := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).Done()
	drain(stream)

	logged := timingLogged(t, logs)
	require.Equal(t, "acme", logged.AppID)
	require.Nil(t, logged.AgentID, "a direct call has no agent to name")
	require.False(t, logged.BYOK, "asked on the deployment's key")
}

func TestTheTimingLogSaysWhyAResponseStoppedEarly(t *testing.T) {
	logs := logTimings(t)
	session, provider, _ := sinkSession(t)

	stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.NoError(t, err)
	provider.script(0).OutputText("a long ans")
	provider.script(0).Incomplete(llm.ReasonMaxOutputTokens)
	drain(stream)

	logged := timingLogged(t, logs)
	require.Equal(t, string(llm.StatusIncomplete), logged.Status)
	require.Equal(t, llm.ReasonMaxOutputTokens, logged.IncompleteReason)
}

func TestTheTimingLogSaysWhatAFailedStreamFailedWith(t *testing.T) {
	for name, test := range map[string]struct {
		err       error
		transient bool
	}{
		"a vendor overloaded": {&llm.ProviderFailure{Err: errors.New("overloaded"), Transient: true}, true},
		"a request refused":   {&llm.ProviderFailure{Err: errors.New("bad request"), Transient: false}, false},
	} {
		t.Run(name, func(t *testing.T) {
			logs := logTimings(t)
			session, provider, _ := sinkSession(t)

			stream, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
			require.NoError(t, err)
			provider.script(0).Fail(test.err, "stream")
			drain(stream)

			logged := timingLogged(t, logs)
			require.Equal(t, "provider_error", logged.ErrorCode)
			require.Contains(t, logged.Error, test.err.Error())
			require.NotNil(t, logged.Transient)
			require.Equal(t, test.transient, *logged.Transient)
		})
	}
}

func TestTheTimingLogSaysWhatAFailedCreateFailedWith(t *testing.T) {
	logs := logTimings(t)
	session, provider, _ := sinkSession(t)
	provider.createErr = &llm.ProviderFailure{Err: errors.New("stub: vendor overloaded"), Transient: true}

	_, err := session.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: prompt()})
	require.Error(t, err)

	logged := timingLogged(t, logs)
	require.Equal(t, "acme", logged.AppID)
	require.Contains(t, logged.Error, "stub: vendor overloaded")
	require.NotNil(t, logged.Transient)
	require.True(t, *logged.Transient)
}

func TestTheTimingLogNamesTheHostAnOpenAICompatibleProviderAsks(t *testing.T) {
	provider, err := Started(openaicompat.New(openaicompat.Options{
		Provider: "gemma", Model: "gemma-4", APIKey: "k",
		BaseURL: "https://user:secret@gemma.example.test:8443/v1",
	}))
	require.NoError(t, err)
	require.Equal(t, "gemma.example.test:8443", vendorHost(provider),
		"the host alone, without the credentials or the path")
	require.Empty(t, vendorHost(newStubLLM()), "a provider that names no host")
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
