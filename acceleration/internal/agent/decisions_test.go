package agent

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/stretchr/testify/suite"
)

type DecisionRecorderSuite struct {
	suite.Suite
}

func TestDecisionRecorderSuite(t *testing.T) {
	suite.Run(t, new(DecisionRecorderSuite))
}

// recorder is one whose writes are whatever send says, not Redis, with a logger that keeps
// what it says.
func (s *DecisionRecorderSuite) recorder(drain time.Duration, send func(context.Context, []live.CallEvent) error) (*decisionRecorder, *bytes.Buffer) {
	var logs bytes.Buffer
	r := buildDecisionRecorder(nil, routing.Owner{CustomerID: "acme", AgentID: "agent-1"}, slog.New(slog.NewTextHandler(&logs, nil)))
	r.drainTimeout = drain
	r.send = send
	return r.start(), &logs
}

func (s *DecisionRecorderSuite) TestTheDrainDeadlineIsTwoWrites() {
	r := newDecisionRecorder(nil, routing.Owner{}, slog.New(slog.DiscardHandler))
	r.Close()

	s.Equal(2*decisionWriteTimout, r.drainTimeout)
}

// TestCloseGivesUpOnARedisThatNeverAnswers is the teardown of a call whose Redis is stuck:
// every batch used to get its own decisionWriteTimout, so closing waited for all of them.
// Now it waits for one deadline and drops what is left, counted once.
func (s *DecisionRecorderSuite) TestCloseGivesUpOnARedisThatNeverAnswers() {
	const queued = 200
	var attempts atomic.Int64
	r, logs := s.recorder(200*time.Millisecond, func(ctx context.Context, _ []live.CallEvent) error {
		attempts.Add(1)
		<-ctx.Done()
		return ctx.Err()
	})
	for i := 0; i < queued; i++ {
		r.Record(Decided{Kind: "reply", TurnID: "turn"})
	}

	closed := make(chan struct{})
	started := time.Now()
	go func() {
		r.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		s.FailNow("Close was still waiting on a Redis that never answers")
	}

	s.GreaterOrEqual(time.Since(started), 200*time.Millisecond, "closing does not drop rows before the deadline")
	s.EqualValues(queued, r.dropped.Load(), "the batch in flight and every queued row are counted")
	s.EqualValues(1, attempts.Load(), "no batch is tried once the deadline has passed")
	s.Equal(1, strings.Count(logs.String(), "dropped decisions"), "the drops are logged once")
	s.Contains(logs.String(), "count=200")
}

// TestCloseWritesEveryQueuedRowWhenRedisAnswers is the control: a healthy Redis loses
// nothing, however many judgements are waiting when the call ends.
func (s *DecisionRecorderSuite) TestCloseWritesEveryQueuedRowWhenRedisAnswers() {
	var mu sync.Mutex
	var written []string
	r, logs := s.recorder(time.Minute, func(_ context.Context, batch []live.CallEvent) error {
		mu.Lock()
		defer mu.Unlock()
		for _, row := range batch {
			written = append(written, row.Reason)
		}
		return nil
	})
	var want []string
	for i := 0; i < 300; i++ {
		reason := strings.Repeat("x", i%5) + string(rune('a'+i%26))
		want = append(want, reason)
		r.Record(Decided{Kind: "reply", Reason: reason})
	}

	r.Close()

	mu.Lock()
	defer mu.Unlock()
	s.Equal(want, written, "every judgement, in the order it was made")
	s.Zero(r.dropped.Load())
	s.NotContains(logs.String(), "dropped")
}
