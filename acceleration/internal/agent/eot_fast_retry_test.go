package agent

import (
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// scores records what the acoustic endpoint was asked, and answers every request with the
// same probability after the given delay.
type scores struct {
	mu           sync.Mutex
	at           []time.Time
	bytes        []int
	inflight     atomic.Int32
	mostInflight atomic.Int32
}

func (r *scores) record(size int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.at = append(r.at, time.Now())
	r.bytes = append(r.bytes, size)
}

func (r *scores) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.at)
}

// sized is how many requests carried exactly this much audio.
func (r *scores) sized(size int) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	var matching int
	for _, got := range r.bytes {
		if got == size {
			matching++
		}
	}
	return matching
}

// grew reports whether each request carried more audio than the one before it.
func (r *scores) grew() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := 1; i < len(r.bytes); i++ {
		if r.bytes[i] <= r.bytes[i-1] {
			return false
		}
	}
	return true
}

func (r *scores) between(first, second int) time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.at[second].Sub(r.at[first])
}

// lowScores serves an acoustic endpoint that always rules the words unfinished.
func (s *AgentSuite) lowScores(delay time.Duration) *scores {
	seen := new(scores)
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read acoustic frame: %v", err)
			return
		}
		inflight := seen.inflight.Add(1)
		defer seen.inflight.Add(-1)
		for most := seen.mostInflight.Load(); inflight > most; most = seen.mostInflight.Load() {
			if seen.mostInflight.CompareAndSwap(most, inflight) {
				break
			}
		}
		seen.record(len(pcm))
		time.Sleep(delay)
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.1)
	}))
	return seen
}

// patience sets how long unfinished words are waited on.
func (s *AgentSuite) patience(patience time.Duration) {
	s.agent.converse.mu.Lock()
	defer s.agent.converse.mu.Unlock()
	s.agent.converse.patience = patience
}

func (s *AgentSuite) TestALowAcousticScoreIsAskedAgainSoonerThanTheUsualRetry() {
	seen := s.lowScores(0)
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return seen.count() == 1 }, "the words were never scored")

	s.speak(participant)

	s.eventually(func() bool { return seen.count() == 2 }, "fresh audio was not scored again")
	s.Less(seen.between(0, 1), 500*time.Millisecond, "the retry waited for the usual pause")
	s.GreaterOrEqual(seen.between(0, 1), primaryEOTLowRetry-20*time.Millisecond)
}

func (s *AgentSuite) TestAudioAlreadyScoredIsNotCopiedOrScoredAgain() {
	seen := s.lowScores(0)
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return seen.count() == 1 }, "the words were never scored")
	s.Never(func() bool { return seen.count() > 1 }, 800*time.Millisecond,
		10*time.Millisecond, "unchanged audio was scored again")
	s.Len(s.model.requests(), 1, "a retry started a second preview")

	s.speak(participant)

	s.eventually(func() bool { return seen.count() == 2 }, "fresh audio was not scored")
	s.Len(s.model.requests(), 1, "a retry started a second preview")
}

func (s *AgentSuite) TestOneScoreIsInFlightAtATimeAndRetriesStartNoPreview() {
	seen := s.lowScores(120 * time.Millisecond)
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	done := make(chan struct{})
	s.T().Cleanup(func() { close(done) })
	go func() {
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				s.speak(participant)
			case <-done:
				return
			}
		}
	}()

	s.eventually(func() bool { return seen.count() >= 4 }, "the words were not scored repeatedly")

	s.EqualValues(1, seen.mostInflight.Load(), "a score was started while another was still running")
	s.True(seen.grew(), "a retry scored audio that was older than what had arrived")
	s.Len(s.model.requests(), 1, "retries churned previews")
	s.LessOrEqual(s.previewsHeld(), 1)
}

func (s *AgentSuite) TestThePatienceForWordsEndsWithNothingNewToScore() {
	seen := s.lowScores(0)
	s.join(false)
	s.patience(400 * time.Millisecond)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}

	s.primaryCandidate(participant, "please find a table")

	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 },
		"the words were never answered once the patience ran out")
	s.EqualValues(1, seen.count(), "nothing new was heard, so nothing was scored again")
	s.Require().Len(s.model.requests(), 2)
	s.Contains(s.model.requests()[1].Instructions, unfinishedNote,
		"the same words are answered with a question, not as if they were finished")
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestNewWordsRestartTheWaitAndAreScoredAtOnce() {
	seen := s.lowScores(0)
	s.join(false)
	s.patience(500 * time.Millisecond)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return seen.count() == 1 }, "the words were never scored")
	time.Sleep(300 * time.Millisecond)

	s.speak(participant)
	s.says(participant, "please find a table for two")

	s.eventually(func() bool { return seen.count() == 2 }, "the new words were not scored")
	// The first words' patience ended at 500 ms. The new words are given their own.
	events := s.events
	s.Never(func() bool { return countOf[Responding](events.seen()) > 0 }, 150*time.Millisecond,
		10*time.Millisecond, "the new words inherited the patience of the old ones")
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 },
		"the new words were never answered once their own patience ran out")
}

func (s *AgentSuite) TestTwoParticipantsAreRetriedOnTheirOwnAudio() {
	seen := s.lowScores(0)
	s.join(false)
	alice := stt.Participant{ID: "alice", UserID: "alice", Name: "Alice"}
	bob := stt.Participant{ID: "bob", UserID: "bob", Name: "Bob"}
	// Each speaks a different amount, which is what tells their requests apart.
	s.speak(alice)
	s.speak(bob)
	s.speak(bob)
	s.speak(bob)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(alice.ID)) == 640 && len(s.agent.eotAudioSnapshot(bob.ID)) == 1920
	}, "the audio was not retained")
	s.says(alice, "please find a table")
	s.says(bob, "what time is it")
	s.eventually(func() bool { return seen.sized(640) == 1 && seen.sized(1920) == 1 },
		"both were not scored")

	s.speak(alice)

	s.eventually(func() bool { return seen.sized(1280) == 1 }, "alice's fresh audio was not scored")
	s.Never(func() bool { return seen.sized(1920) > 1 }, 600*time.Millisecond, 10*time.Millisecond,
		"bob was scored again on audio he had not added to")
}

func (s *AgentSuite) TestWhenTheFloorChangesTheRetryIsNotScoredByEar() {
	seen := s.lowScores(0)
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return seen.count() == 1 && s.keptPreviews() == 1 },
		"the words were never scored")
	s.agent.mu.Lock()
	s.agent.generating = true
	s.agent.mu.Unlock()
	s.T().Cleanup(func() {
		s.agent.mu.Lock()
		s.agent.generating = false
		s.agent.mu.Unlock()
	})

	s.speak(participant)

	s.eventually(func() bool { return len(s.flow.requests()) >= 1 },
		"the retry was not put to the flow controller while the agent held the floor")
	s.Equal(1, seen.count(), "an acoustic score was asked for while the agent was talking")
	s.eventually(func() bool { return s.keptPreviews() == 0 }, "the kept preview survived the floor changing")
}

func (s *AgentSuite) TestClosingStopsTheRetries() {
	seen := s.lowScores(0)
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return seen.count() == 1 }, "the words were never scored")

	// Fresh audio is waiting to be scored when the retry comes due.
	s.speak(participant)

	s.Require().NoError(s.agent.Close())

	events := s.events
	s.Never(func() bool { return seen.count() > 1 || countOf[Responding](events.seen()) > 0 }, 600*time.Millisecond,
		10*time.Millisecond, "a closed agent kept retrying")
	s.Zero(s.previewsHeld())
	s.Zero(s.keptPreviews())
}
