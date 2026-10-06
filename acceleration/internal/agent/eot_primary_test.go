package agent

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (s *AgentSuite) primaryEOTServer(handler http.Handler) *httptest.Server {
	s.eotMode = EOTModePrimary
	server := httptest.NewServer(handler)
	s.T().Cleanup(server.Close)
	client, err := NewEOTClient(server.URL, "")
	s.Require().NoError(err)
	s.eot = client
	return server
}

func (s *AgentSuite) primaryCandidate(participant stt.Participant, text string) {
	s.speak(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	s.says(participant, text)
}

func primaryScoreHandler(t *AgentSuite, score float64, requests *atomic.Int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			t.T().Errorf("read EOT audio: %v", err)
			return
		}
		writeEOTResponse(t.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, score)
	}
}

func (s *AgentSuite) TestPrimaryEOTHighAnswersWithoutCallingSemanticController() {
	var requests atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, 0.9, &requests))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"a high primary EOT score did not release the caller turn")
	s.EqualValues(1, requests.Load())
	s.Empty(s.flow.requests(), "a valid high EOT score must bypass the semantic flow model")
	s.Contains(said(s.voice.spoken()), "Hello there.")
}

func (s *AgentSuite) TestPrimaryEOTLowWaitsWithoutCallingSemanticController() {
	var requests atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, 0.1, &requests))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")

	s.eventually(func() bool {
		s.agent.converse.mu.Lock()
		defer s.agent.converse.mu.Unlock()
		_, waiting := s.agent.converse.waiting[participant.ID]
		return waiting
	}, "a low primary EOT score did not leave the caller turn pending")
	s.Empty(s.flow.requests(), "a valid low EOT score must bypass the semantic flow model")
	s.Empty(s.voice.spoken())
	s.Zero(countOf[Responded](s.reported()))
	s.EqualValues(1, requests.Load(), "a valid low endpoint probability is a final score, not a transient failure")
}

func (s *AgentSuite) TestPrimaryEOTRetriesTransientFailureThenUsesHighScore() {
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requests.Add(1)
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read EOT audio: %v", err)
			return
		}
		if count == 1 {
			http.Error(w, "temporary", http.StatusServiceUnavailable)
			return
		}
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.9)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"a successful retry did not resolve the caller turn")
	s.EqualValues(2, requests.Load())
	s.Empty(s.flow.requests(), "a successful high score after a retry must not use semantic flow")
}

func (s *AgentSuite) TestPrimaryEOTRetriesTwiceThenTreatsLowScoreAsWait() {
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requests.Add(1)
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read EOT audio: %v", err)
			return
		}
		if count <= 2 {
			http.Error(w, "temporary", http.StatusServiceUnavailable)
			return
		}
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.1)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")

	s.eventually(func() bool {
		s.agent.converse.mu.Lock()
		defer s.agent.converse.mu.Unlock()
		_, waiting := s.agent.converse.waiting[participant.ID]
		return waiting
	}, "a valid low score following two transient failures did not leave the caller waiting")
	s.EqualValues(3, requests.Load())
	s.Empty(s.flow.requests())
	s.Zero(countOf[Responded](s.reported()))
	s.Empty(s.voice.spoken())
}

func (s *AgentSuite) TestPrimaryEOTExhaustsBoundedTransientRetriesBeforeSemanticFallback() {
	started := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseEOT := func() { releaseOnce.Do(func() { close(release) }) }
	s.T().Cleanup(releaseEOT)
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requests.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		if count == 1 {
			close(started)
			<-release
		}
		http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	select {
	case <-started:
	case <-time.After(settleFor):
		s.FailNow("the primary EOT request did not start")
	}
	s.Empty(s.flow.requests(), "primary mode must not start semantic work before EOT fails")
	releaseEOT()
	s.eventually(func() bool { return len(s.flow.requests()) == 1 && requests.Load() == 3 },
		"semantic fallback did not wait for all three bounded transient attempts")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"semantic fallback did not answer after EOT failed")
	s.EqualValues(3, requests.Load(), "primary mode makes one initial request and at most two transient retries")
}

func (s *AgentSuite) TestPrimaryEOTWithoutAudioFallsBackToSemanticController() {
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), eotMinSamples, 0.9)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	// Open the participant's transcription listener, then clear the setup audio so this
	// candidate exercises the no-window fallback rather than the score endpoint.
	s.speak(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	s.agent.mu.Lock()
	ring := s.agent.audioHistory[participant.ID]
	s.agent.mu.Unlock()
	ring.clear()
	s.says(participant, "please find a table")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"a candidate without an audio window did not use semantic fallback")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"semantic fallback without audio did not answer")
	s.Zero(requests.Load())
}

func (s *AgentSuite) primaryLateFloorChange(score float64) {
	started := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseEOT := func() { releaseOnce.Do(func() { close(release) }) }
	s.T().Cleanup(releaseEOT)
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read EOT audio: %v", err)
			return
		}
		close(started)
		<-release
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, score)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "please find a table")
	select {
	case <-started:
	case <-time.After(settleFor):
		s.FailNow("the primary EOT request did not start")
	}
	s.Empty(s.flow.requests(), "primary mode must wait for EOT before semantic fallback")

	s.agent.mu.Lock()
	s.agent.generating = true
	s.agent.saying = "the updated reply"
	s.agent.history = append(s.agent.history, llm.Message{Role: llm.Assistant, Content: "the updated reply"})
	s.agent.mu.Unlock()
	releaseEOT()
	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"a score received after the floor changed did not use semantic fallback")
	question := s.flow.requests()[0].Input[0].Content
	s.Contains(question, "is speaking right now")
	s.Contains(question, `has so far said "the updated reply"`)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the refreshed semantic fallback did not resolve the caller turn")
}

func (s *AgentSuite) TestPrimaryEOTLateHighScoreUsesRefreshedSemanticFloorState() {
	s.primaryLateFloorChange(0.9)
}

func (s *AgentSuite) TestPrimaryEOTLateLowScoreUsesRefreshedSemanticFloorState() {
	s.primaryLateFloorChange(0.1)
}

func (s *AgentSuite) TestPrimaryEOTSkipsScoringWhenTheFloorIsAlreadyActive() {
	var requests atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, 0.9, &requests))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.speak(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	s.agent.mu.Lock()
	s.agent.generating = true
	s.agent.saying = "the current reply"
	s.agent.mu.Unlock()
	s.says(participant, "please stop and listen")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"an active-floor candidate did not use the semantic controller")
	s.Zero(requests.Load(), "EOT must not score a candidate that arrives during active speech")
	s.Contains(s.flow.requests()[0].Input[0].Content, "is speaking right now")
}

func (s *AgentSuite) TestPrimaryEOTRevisionCancelsOldCandidateAndDoesNotCallSemanticController() {
	firstStarted := make(chan struct{})
	firstCanceled := make(chan struct{})
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := requests.Add(1)
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read EOT audio: %v", err)
			return
		}
		if count == 1 {
			close(firstStarted)
			<-r.Context().Done()
			close(firstCanceled)
			return
		}
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.9)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.primaryCandidate(participant, "could you find a table")
	select {
	case <-firstStarted:
	case <-time.After(settleFor):
		s.FailNow("the first primary EOT request did not start")
	}
	s.mutters(participant, "could you find a table for four")
	select {
	case <-firstCanceled:
	case <-time.After(settleFor):
		s.FailNow("a transcript revision did not cancel the old primary EOT request")
	}
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the revised candidate did not resolve from its high EOT score")
	s.EqualValues(2, requests.Load(), "the revision should replace rather than duplicate the candidate")
	s.Empty(s.flow.requests(), "primary high scores must not invoke the semantic controller")
}

func (s *AgentSuite) TestPrimaryEOTLateDiarizationUsesCurrentSpeakerForFallback() {
	started := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseEOT := func() { releaseOnce.Do(func() { close(release) }) }
	s.T().Cleanup(releaseEOT)
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read EOT audio: %v", err)
			return
		}
		close(started)
		<-release
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.9)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.speak(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the participant audio window was not retained")
	s.saysInVoice(participant, "please find a table", "caller-voice")
	select {
	case <-started:
	case <-time.After(settleFor):
		s.FailNow("the primary EOT request did not start")
	}

	// The transcriber can add diarization to the same settled words without revising the
	// candidate ID. The pending score must use this current speaker instead of the snapshot
	// that originally started the request.
	s.saysInVoice(participant, "please find a table", "another-voice")
	var ready candidate
	s.eventually(func() bool {
		s.agent.mu.Lock()
		for _, gate := range s.agent.eotGates {
			if gate.participantID == participant.ID {
				ready = gate.ready
				break
			}
		}
		s.agent.mu.Unlock()
		if ready.ID == "" {
			return false
		}
		current, ok := s.agent.cadence.candidateSnapshot(ready)
		return ok && current.Speaker == "another-voice"
	}, "the same candidate did not observe the late speaker update")
	releaseEOT()
	s.eventually(func() bool { return len(s.flow.requests()) == 1 },
		"a same-text diarization change did not use the semantic controller")
	s.Contains(s.flow.requests()[0].Input[0].Content, "in a different voice")
}
