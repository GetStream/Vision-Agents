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
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/stretchr/testify/suite"
)

type TurnRecorderSuite struct {
	suite.Suite
}

func TestTurnRecorderSuite(t *testing.T) {
	suite.Run(t, new(TurnRecorderSuite))
}

// recorder is one with nowhere to write. Nothing here lets a row reach the writer, so the
// store it would need is never touched.
func (s *TurnRecorderSuite) recorder() *turnRecorder {
	return newTurnRecorder(nil, nil, routing.Owner{CustomerID: "acme"}, slog.New(slog.DiscardHandler))
}

// TestATurnRecordedAfterCloseIsLetGo covers the one send that cannot be recovered from.
//
// Interrupting a session closes the recorder and then reports the turn it cut short, in
// that order, and a send on a closed channel panics even from a select with a default --
// so this arrived as a panic that took the whole router down rather than as a lost row.
func (s *TurnRecorderSuite) TestATurnRecordedAfterCloseIsLetGo() {
	r := s.recorder()
	r.Close()

	s.NotPanics(func() {
		r.Record(Turn{TurnID: "turn-1", StartedAt: time.Now().UTC()})
	})
}

// TestClosingTwiceIsHarmless because teardown reaches it from more than one direction.
func (s *TurnRecorderSuite) TestClosingTwiceIsHarmless() {
	r := s.recorder()

	s.NotPanics(func() {
		r.Close()
		r.Close()
		r.Record(Turn{TurnID: "turn-2", StartedAt: time.Now().UTC()})
	})
}

// TestTurnsRecordedWhileClosingAreLetGo is the race itself, which is how it happened: the
// turn was reported from the socket's goroutine while the session was being torn down.
func (s *TurnRecorderSuite) TestTurnsRecordedWhileClosingAreLetGo() {
	r := s.recorder()
	r.Close()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for n := 0; n < 50; n++ {
				r.Record(Turn{TurnID: "turn-3", StartedAt: time.Now().UTC()})
			}
		}()
	}

	s.NotPanics(wg.Wait)
}

// slowRecorder is one whose writes are whatever write says, not a store, with a logger
// that keeps what it says.
func (s *TurnRecorderSuite) slowRecorder(drain time.Duration, write func(context.Context, live.Turn) error) (*turnRecorder, *bytes.Buffer) {
	var logs bytes.Buffer
	r := buildTurnRecorder(nil, nil, routing.Owner{CustomerID: "acme", AgentID: "agent-1"}, slog.New(slog.NewTextHandler(&logs, nil)))
	r.drainTimeout = drain
	r.write = write
	return r.start(), &logs
}

func (s *TurnRecorderSuite) TestTheDrainDeadlineIsTwoWrites() {
	r := s.recorder()
	r.Close()

	s.Equal(2*turnWriteTimeout, r.drainTimeout)
}

// TestCloseGivesUpOnAStoreThatNeverAnswers is the teardown of a call whose database is
// stuck: every queued row used to get its own turnWriteTimeout, so closing waited for all
// of them. Now it waits for one deadline and drops what is left, counted once.
func (s *TurnRecorderSuite) TestCloseGivesUpOnAStoreThatNeverAnswers() {
	const queued = 20
	var attempts atomic.Int64
	r, logs := s.slowRecorder(200*time.Millisecond, func(ctx context.Context, _ live.Turn) error {
		attempts.Add(1)
		<-ctx.Done()
		return ctx.Err()
	})
	for i := 0; i < queued; i++ {
		r.Record(Turn{TurnID: "turn", StartedAt: time.Now().UTC()})
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
		s.FailNow("Close was still waiting on a store that never answers")
	}

	s.GreaterOrEqual(time.Since(started), 200*time.Millisecond, "closing does not drop rows before the deadline")
	s.EqualValues(queued, r.dropped.Load(), "the row in flight and every queued one are counted")
	s.EqualValues(1, attempts.Load(), "no row is tried once the deadline has passed")
	s.Equal(1, strings.Count(logs.String(), "dropped turns"), "the drops are logged once")
	s.Contains(logs.String(), "count=20")
}

// TestCloseWritesEveryQueuedRowWhenTheStoreAnswers is the control: a healthy store loses
// nothing, however many rows are waiting when the call ends.
func (s *TurnRecorderSuite) TestCloseWritesEveryQueuedRowWhenTheStoreAnswers() {
	var mu sync.Mutex
	var written []string
	r, logs := s.slowRecorder(time.Minute, func(_ context.Context, row live.Turn) error {
		time.Sleep(time.Millisecond)
		mu.Lock()
		defer mu.Unlock()
		written = append(written, row.TurnID)
		return nil
	})
	var want []string
	for i := 0; i < turnQueueSize; i++ {
		id := "turn-" + strings.Repeat("x", i%7) + string(rune('a'+i%26))
		want = append(want, id)
		r.Record(Turn{TurnID: id, StartedAt: time.Now().UTC()})
	}

	r.Close()

	mu.Lock()
	defer mu.Unlock()
	s.Equal(want, written, "every row, in the order it finished")
	s.Zero(r.dropped.Load())
	s.NotContains(logs.String(), "dropped")
	s.Error(r.ctx.Err(), "the writer's context is released")
}

// TestAWriteThatFinishesPastTheDeadlineIsNotCounted keeps the count to rows that were lost:
// the one in flight when the deadline passes is stored, the two behind it are dropped.
func (s *TurnRecorderSuite) TestAWriteThatFinishesPastTheDeadlineIsNotCounted() {
	var attempts atomic.Int64
	r, _ := s.slowRecorder(50*time.Millisecond, func(context.Context, live.Turn) error {
		attempts.Add(1)
		time.Sleep(300 * time.Millisecond)
		return nil
	})
	for i := 0; i < 3; i++ {
		r.Record(Turn{TurnID: "turn", StartedAt: time.Now().UTC()})
	}

	r.Close()

	s.EqualValues(1, attempts.Load())
	s.EqualValues(2, r.dropped.Load())
}

// TestAWriteThatTimesOutOnItsOwnIsCounted keeps the warning's count equal to the rows lost
// in a stuck store: the first rows give up on their own timeout, before any deadline.
func (s *TurnRecorderSuite) TestAWriteThatTimesOutOnItsOwnIsCounted() {
	var logs bytes.Buffer
	r := buildTurnRecorder(nil, nil, routing.Owner{CustomerID: "acme"}, slog.New(slog.NewTextHandler(&logs, nil)))
	r.writeTimeout = 20 * time.Millisecond
	r.write = func(ctx context.Context, _ live.Turn) error {
		<-ctx.Done()
		return ctx.Err()
	}
	r.start()
	for i := 0; i < 3; i++ {
		r.Record(Turn{TurnID: "turn", StartedAt: time.Now().UTC()})
	}

	r.Close()

	s.EqualValues(3, r.dropped.Load())
	s.Contains(logs.String(), "count=3")
}

// TestAFailedWriteIsNotADrop keeps the count to what the deadline cut: a store that
// answers with an error is logged by the writer, and the call is not behind.
func (s *TurnRecorderSuite) TestAFailedWriteIsNotADrop() {
	r, _ := s.slowRecorder(time.Minute, func(context.Context, live.Turn) error {
		return context.DeadlineExceeded
	})
	for i := 0; i < 5; i++ {
		r.Record(Turn{TurnID: "turn", StartedAt: time.Now().UTC()})
	}

	r.Close()

	s.Zero(r.dropped.Load())
}

func (s *TurnRecorderSuite) TestTurnTimingStartsAtLastTranscriptRevision() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var finished Turn
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, func(turn Turn) { finished = turn })
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	tracker.modelStarted("turn-1", base.Add(400*time.Millisecond))
	tracker.firstText("turn-1", base.Add(700*time.Millisecond))
	tracker.ttsStarted("turn-1", base.Add(740*time.Millisecond))
	tracker.firstAudio("turn-1", base.Add(900*time.Millisecond))
	tracker.spoke("turn-1", 160, 500)
	tracker.completed("turn-1", 280, 1)

	s.Equal(base, finished.StartedAt)
	s.InDelta(350, finished.CadenceMs, 0.001)
	s.InDelta(50, finished.DecisionMs, 0.001)
	s.InDelta(300, finished.ModelToFirstTextMs, 0.001)
	s.InDelta(40, finished.TextToTTSMs, 0.001)
	s.InDelta(160, finished.TTSToAudioMs, 0.001)
	s.InDelta(900, finished.RoundtripMs, 0.001)
	s.InDelta(1020, finished.SpeechEndToAudioMs, 0.001)
}

// reported collects the turns a tracker closes, safe to read while it is still closing them.
type reported struct {
	mu    sync.Mutex
	turns []Turn
}

func (r *reported) add(turn Turn) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.turns = append(r.turns, turn)
}

func (r *reported) all() []Turn {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Turn(nil), r.turns...)
}

// spokenTurn opens a turn and runs it up to the end of its speech, with publishing having
// returned at the given offset from the last transcript revision.
func spokenTurn(tracker *turnTracker, base time.Time, returned time.Duration) {
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	tracker.modelStarted("turn-1", base.Add(400*time.Millisecond))
	tracker.firstText("turn-1", base.Add(700*time.Millisecond))
	tracker.ttsStarted("turn-1", base.Add(740*time.Millisecond))
	tracker.firstAudio("turn-1", base.Add(returned))
	tracker.spoke("turn-1", 160, 500)
	tracker.completed("turn-1", 280, 1)
}

func (s *TurnRecorderSuite) TestTurnTimingCarriesWhenTheEdgeQueuedAndPlayedTheFirstFrame() {
	// Publishing a long first chunk returns only as the track drains it, so the return is
	// later than both moments the edge reports.
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	marks := tracker.marksFor("turn-1")
	s.Require().NotNil(marks)

	marks.FirstFrameQueued(base.Add(880 * time.Millisecond))
	marks.FirstAudiblePulled(base.Add(940 * time.Millisecond))
	tracker.firstAudio("turn-1", base.Add(1400*time.Millisecond))
	tracker.spoke("turn-1", 160, 500)
	tracker.completed("turn-1", 280, 1)

	turns := done.all()
	s.Require().Len(turns, 1)
	s.InDelta(880, turns[0].FirstFrameQueuedMs, 0.001)
	s.InDelta(940, turns[0].FirstAudibleFrameMs, 0.001)
	s.InDelta(1060, turns[0].SpeechEndToAudibleMs, 0.001, "the speech-end figure adds the transcriber's wait")
	s.InDelta(1400, turns[0].RoundtripMs, 0.001, "the figure measured at the return is kept")
	s.InDelta(1520, turns[0].SpeechEndToAudioMs, 0.001)
	s.Less(turns[0].FirstFrameQueuedMs, turns[0].RoundtripMs)
}

func (s *TurnRecorderSuite) TestAnEdgeThatReportsNothingLeavesTheNewMomentsEmpty() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)

	spokenTurn(tracker, base, 900*time.Millisecond)

	turns := done.all()
	s.Require().Len(turns, 1, "a turn with no reports from the edge is not held back")
	s.Zero(turns[0].FirstFrameQueuedMs)
	s.Zero(turns[0].FirstAudibleFrameMs)
	s.Zero(turns[0].SpeechEndToAudibleMs)
}

func (s *TurnRecorderSuite) TestATurnWaitsForTheTrackToTakeItsQueuedSpeech() {
	// A short reply is queued whole before the track has taken a frame, so the synthesis
	// is complete first. Closing the turn then would report it without the moment that was
	// about to arrive.
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	marks := tracker.marksFor("turn-1")
	marks.FirstFrameQueued(base.Add(900 * time.Millisecond))
	tracker.firstAudio("turn-1", base.Add(905*time.Millisecond))
	tracker.spoke("turn-1", 160, 300)
	tracker.completed("turn-1", 280, 1)

	s.Empty(done.all(), "the turn closed before the track took its speech")

	marks.FirstAudiblePulled(base.Add(920 * time.Millisecond))

	s.Require().Eventually(func() bool { return len(done.all()) == 1 }, time.Second, time.Millisecond)
	s.InDelta(920, done.all()[0].FirstAudibleFrameMs, 0.001)
}

func (s *TurnRecorderSuite) TestTheTrackNeverWaitsOnWhoeverConsumesTheTurn() {
	// The report runs on the track's clock. Closing the turn there would put an event
	// consumer's backpressure in the middle of the audio.
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	release := make(chan struct{})
	var once sync.Once
	s.T().Cleanup(func() { once.Do(func() { close(release) }) })
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, func(turn Turn) {
		<-release
		done.add(turn)
	})
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	marks := tracker.marksFor("turn-1")
	marks.FirstFrameQueued(base.Add(900 * time.Millisecond))
	tracker.spoke("turn-1", 160, 300)
	tracker.completed("turn-1", 280, 1)

	returned := make(chan struct{})
	go func() {
		marks.FirstAudiblePulled(base.Add(920 * time.Millisecond))
		close(returned)
	}()

	select {
	case <-returned:
	case <-time.After(time.Second):
		s.Fail("reporting the pull waited on the turn's consumer")
	}
	once.Do(func() { close(release) })
	s.Require().Eventually(func() bool { return len(done.all()) == 1 }, time.Second, time.Millisecond)
}

func (s *TurnRecorderSuite) TestATurnIsReportedWithoutAMomentTheTrackNeverReached() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)
	tracker.grace = 20 * time.Millisecond
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	tracker.marksFor("turn-1").FirstFrameQueued(base.Add(900 * time.Millisecond))
	tracker.spoke("turn-1", 160, 300)
	tracker.completed("turn-1", 280, 1)

	s.Require().Eventually(func() bool { return len(done.all()) == 1 }, time.Second, time.Millisecond,
		"a track that stopped pulling must not hold the turn forever")

	turn := done.all()[0]
	s.InDelta(900, turn.FirstFrameQueuedMs, 0.001)
	s.Zero(turn.FirstAudibleFrameMs, "the moment that never happened is left out")
}

func (s *TurnRecorderSuite) TestAnInterruptionClosesATurnWaitingOnTheTrack() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	marks := tracker.marksFor("turn-1")
	marks.FirstFrameQueued(base.Add(900 * time.Millisecond))
	tracker.spoke("turn-1", 160, 300)
	tracker.completed("turn-1", 280, 1)

	tracker.interrupt("turn-1", interruption{})
	marks.FirstAudiblePulled(base.Add(920 * time.Millisecond))
	time.Sleep(10 * time.Millisecond)

	turns := done.all()
	s.Require().Len(turns, 1, "a turn closes exactly once")
	s.True(turns[0].Interrupted)
	s.Zero(turns[0].FirstAudibleFrameMs, "speech heard after the turn was closed is not part of it")
}

func (s *TurnRecorderSuite) TestAReportThatOutlivesItsTurnIsDropped() {
	// The marks belong to one turn, so what an edge says late about the speech it was given
	// for a closed turn cannot stamp the one that followed.
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var done reported
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, done.add)
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	stale := tracker.marksFor("turn-1")
	tracker.interrupt("turn-1", interruption{})
	tracker.begin("turn-2", stt.Participant{}, base.Add(5350*time.Millisecond), base.Add(5*time.Second), 120)

	stale.FirstFrameQueued(base.Add(5900 * time.Millisecond))
	stale.FirstAudiblePulled(base.Add(5920 * time.Millisecond))
	tracker.spoke("turn-2", 160, 300)
	tracker.completed("turn-2", 280, 1)

	turns := done.all()
	s.Require().Len(turns, 2)
	s.Zero(turns[1].FirstFrameQueuedMs)
	s.Zero(turns[1].FirstAudibleFrameMs)
}

func (s *TurnRecorderSuite) TestMarksAreOnlyHandedOutWhileTheTurnStillWantsThem() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	tracker := newTurnTracker(routing.Owner{AgentID: "agent-1", CallID: "call-1"}, func(Turn) {})
	s.Nil(tracker.marksFor("nobody"), "an unmeasured turn has nothing to report to")
	tracker.begin("turn-1", stt.Participant{}, base.Add(350*time.Millisecond), base, 120)
	marks := tracker.marksFor("turn-1")
	s.NotNil(marks)

	marks.FirstAudiblePulled(base.Add(900 * time.Millisecond))

	s.Nil(tracker.marksFor("turn-1"), "once the speech has been heard, the edge has no more to say")
}
