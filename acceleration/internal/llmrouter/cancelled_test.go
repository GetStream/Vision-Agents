package llmrouter

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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
	require.GreaterOrEqual(t, row.LatencyMs, float64(headerWait.Milliseconds()),
		"the first token did arrive, so what the caller waited for it is known")
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
	require.GreaterOrEqual(t, row.LatencyMs, float64(headerWait.Milliseconds()))
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
