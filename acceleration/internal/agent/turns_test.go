package agent

import (
	"log/slog"
	"sync"
	"testing"
	"time"

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
	return newTurnRecorder(nil, routing.Owner{CustomerID: "acme"}, slog.New(slog.DiscardHandler))
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

func (s *TurnRecorderSuite) TestTurnTimingStartsAtLastTranscriptRevision() {
	base := time.Date(2026, 9, 28, 10, 0, 0, 0, time.UTC)
	var finished Turn
	tracker := newTurnTracker(func(turn Turn) { finished = turn })
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
