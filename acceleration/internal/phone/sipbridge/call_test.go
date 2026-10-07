package sipbridge

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func newTestCall() (*Call, *fakeLeg, *fakeLeg) {
	customer, stream := &fakeLeg{}, &fakeLeg{}
	c := newCall(customer, stream, slog.New(slog.DiscardHandler))
	c.establish()
	return c, customer, stream
}

func requireEnded(t *testing.T, c *Call) {
	t.Helper()
	select {
	case <-c.Done():
	default:
		t.Fatal("call has not ended")
	}
}

func TestByeFromCustomerEndsStreamLeg(t *testing.T) {
	c, customer, stream := newTestCall()

	c.onBye(t.Context(), customerSide)

	require.Equal(t, []string{"bye"}, stream.calls())
	require.Empty(t, customer.calls())
	requireEnded(t, c)
	require.NoError(t, c.Err())
}

func TestByeFromStreamEndsCustomerLeg(t *testing.T) {
	c, customer, stream := newTestCall()

	c.onBye(t.Context(), streamSide)

	require.Equal(t, []string{"bye"}, customer.calls())
	require.Empty(t, stream.calls())
	requireEnded(t, c)
}

func TestByeFromBothSidesAtOnceEndsCallOnce(t *testing.T) {
	c, customer, stream := newTestCall()

	var wg sync.WaitGroup
	for _, s := range []side{customerSide, streamSide} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.onBye(t.Context(), s)
		}()
	}
	wg.Wait()

	requireEnded(t, c)
	require.NoError(t, c.Err())
	// Exactly one bye should be sent across both legs
	totalByes := len(customer.calls()) + len(stream.calls())
	require.Equal(t, 1, totalByes, "expected exactly one bye sent, got %d", totalByes)
}

func TestAFailedByeStillEndsTheCall(t *testing.T) {
	c, _, stream := newTestCall()
	stream.byeErr = errors.New("connection reset")

	c.onBye(t.Context(), customerSide)

	requireEnded(t, c)
}

func TestReinviteIsForwardedToTheOtherLeg(t *testing.T) {
	c, _, stream := newTestCall()
	stream.reinviteAnswer = []byte("answer")

	got, err := c.onReinvite(t.Context(), customerSide, []byte("offer"))

	require.NoError(t, err)
	require.Equal(t, []byte("answer"), got)
	require.Equal(t, []string{"reinvite"}, stream.calls())
	require.Equal(t, []byte("offer"), stream.body(0))
}

func TestInfoIsForwardedToTheOtherLeg(t *testing.T) {
	c, customer, _ := newTestCall()

	err := c.onInfo(t.Context(), streamSide, "application/dtmf-relay", []byte("Signal=5"))

	require.NoError(t, err)
	require.Equal(t, []string{"info:application/dtmf-relay"}, customer.calls())
	require.Equal(t, []byte("Signal=5"), customer.body(0))
}

func TestLostLegHangsUpTheOtherAndSaysWhy(t *testing.T) {
	c, customer, _ := newTestCall()

	c.lost(t.Context(), streamSide, errors.New("EOF"))

	require.Equal(t, []string{"bye"}, customer.calls())
	requireEnded(t, c)
	require.ErrorContains(t, c.Err(), "stream leg lost: EOF")
}

func TestHangupByesBothLegs(t *testing.T) {
	c, customer, stream := newTestCall()

	require.NoError(t, c.Hangup(t.Context()))

	require.Equal(t, []string{"bye"}, customer.calls())
	require.Equal(t, []string{"bye"}, stream.calls())
	requireEnded(t, c)
}

func TestHangupByesStreamWhileTheCustomerTrunkDoesNotAnswer(t *testing.T) {
	c, customer, stream := newTestCall()
	customer.byeBlocks = true
	ctx, cancel := context.WithCancel(t.Context())
	hungUp := make(chan error, 1)

	go func() { hungUp <- c.Hangup(ctx) }()

	require.Eventually(t, func() bool {
		return len(stream.calls()) == 1 && len(customer.calls()) == 1
	}, time.Second, time.Millisecond)
	require.Equal(t, []error{nil}, stream.byeContextErrs())
	select {
	case <-hungUp:
		t.Fatal("Hangup returned while the customer BYE was still waiting")
	default:
	}
	cancel()
	require.ErrorIs(t, <-hungUp, context.Canceled)
	requireEnded(t, c)
}

func TestHangupBoundsEachByeByItsOwnTimeout(t *testing.T) {
	c, customer, stream := newTestCall()
	var deadlines [2]time.Time
	customer.onBye = func(ctx context.Context) { deadlines[customerSide], _ = ctx.Deadline() }
	stream.onBye = func(ctx context.Context) { deadlines[streamSide], _ = ctx.Deadline() }
	start := time.Now()

	require.NoError(t, c.Hangup(t.Context()))

	for s, deadline := range deadlines {
		require.WithinDuration(t, start.Add(cleanupTimeout), deadline, time.Second, side(s))
	}
}

func TestNothingIsForwardedAfterTheCallEnded(t *testing.T) {
	c, customer, stream := newTestCall()
	c.onBye(t.Context(), customerSide)

	c.onBye(t.Context(), streamSide)

	require.Empty(t, customer.calls())
	require.Equal(t, []string{"bye"}, stream.calls())
}

func TestHangupAfterRemoteByeSendsNothing(t *testing.T) {
	c, customer, stream := newTestCall()

	c.onBye(t.Context(), customerSide)
	err := c.Hangup(t.Context())

	require.NoError(t, err)
	require.Equal(t, []string{"bye"}, stream.calls())
	require.Empty(t, customer.calls())
	requireEnded(t, c)
}

func TestReinviteAfterTheCallEndedIsRefused(t *testing.T) {
	c, customer, stream := newTestCall()
	c.onBye(t.Context(), customerSide)

	got, err := c.onReinvite(t.Context(), streamSide, []byte("offer"))

	require.ErrorIs(t, err, errCallEnded)
	require.Nil(t, got)
	// stream got the bye, customer should get nothing
	require.Equal(t, []string{"bye"}, stream.calls())
	require.Empty(t, customer.calls())
}

func TestInfoAfterTheCallEndedIsRefused(t *testing.T) {
	c, customer, stream := newTestCall()
	c.onBye(t.Context(), streamSide)

	err := c.onInfo(t.Context(), customerSide, "application/dtmf-relay", []byte("Signal=5"))

	require.ErrorIs(t, err, errCallEnded)
	// customer got the bye, stream should get nothing
	require.Equal(t, []string{"bye"}, customer.calls())
	require.Empty(t, stream.calls())
}

// setupCall is a call still being set up: Stream answered, but the customer leg is a real
// leg whose INVITE has no answer yet, so it has no dialog to send requests in.
func setupCall() *Call {
	customer := &sipLeg{side: customerSide, log: slog.New(slog.DiscardHandler)}
	return newCall(customer, &fakeLeg{}, slog.New(slog.DiscardHandler))
}

func TestAReinviteDuringSetupIsAskedToWait(t *testing.T) {
	c := setupCall()
	forward := func(ctx context.Context, offer []byte) ([]byte, error) {
		return c.onReinvite(ctx, streamSide, offer)
	}

	res := answerReinvite(t.Context(), reinviteLeg(), reinviteFrom([]byte("v=0 hold\r\n")), forward)

	require.Equal(t, 491, res.StatusCode)
	require.Equal(t, "Request Pending", res.Reason)
}

func TestAnInfoDuringSetupIsRefused(t *testing.T) {
	c := setupCall()

	err := c.onInfo(t.Context(), streamSide, "application/dtmf-relay", []byte("Signal=5"))

	require.ErrorIs(t, err, errNotEstablished)
	code, reason := infoStatus(err)
	require.Equal(t, 500, code)
	require.Equal(t, "Server Internal Error", reason)
}
