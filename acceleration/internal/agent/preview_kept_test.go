package agent

import (
	"io"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// keptPreviews is how many previews a Wait is holding for the next check of the same words.
func (s *AgentSuite) keptPreviews() int {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	return len(s.agent.kept)
}

// quickRetry has a Wait put the same words again after a moment instead of the usual pause.
func (s *AgentSuite) quickRetry() {
	s.agent.cadence.mu.Lock()
	defer s.agent.cadence.mu.Unlock()
	s.agent.cadence.retry = 40 * time.Millisecond
}

// noRetry stops a Wait putting its words again, so what is kept stays kept.
func (s *AgentSuite) noRetry() {
	s.agent.cadence.mu.Lock()
	defer s.agent.cadence.mu.Unlock()
	s.agent.cadence.retry = time.Hour
}

// watchPreview reports whether the only preview there is has been cancelled.
func (s *AgentSuite) watchPreview() *atomic.Bool {
	cancelled := new(atomic.Bool)
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	s.Require().Len(s.agent.previews, 1, "there is not exactly one preview to watch")
	for _, p := range s.agent.previews {
		cancel := p.cancel
		p.cancel = func() {
			cancelled.Store(true)
			cancel()
		}
	}
	return cancelled
}

// waitsThenAnswers has the flow controller wait on the words once and answer the next time.
func (s *AgentSuite) waitsThenAnswers() {
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"respond","floor":"continue"}`}
}

func (s *AgentSuite) TestAWaitKeepsThePreviewAndTheNextCheckOfTheSameWordsAdoptsIt() {
	s.join(true)
	s.waitsThenAnswers()
	s.quickRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please find a table")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the words were never answered after the wait")
	s.Len(s.flow.requests(), 2, "the words were checked twice")
	s.Len(s.model.requests(), 1, "the reply started for the first check is the one that was spoken")
	s.Contains(said(s.voice.spoken()), "Hello there.")
	s.Zero(s.previewsHeld(), "the answer took the preview")
	s.Zero(s.keptPreviews())
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	responding, _ := firstOf[Responding](s.reported())
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(responding.TurnID, turn.TurnID, "the reply the preview made is the one the turn is about")
}

func (s *AgentSuite) TestAModelCallForAKeptPreviewCountsTowardsTheTurnThatTookItOver() {
	s.join(true)
	s.agent.turns.begin("answer", stt.Participant{ID: "alice"}, time.Now(), time.Now(), 0)
	s.agent.mu.Lock()
	s.agent.previewTurns["preview"] = "answer"
	s.agent.mu.Unlock()

	s.agent.recordModelCall(llm.CallTiming{Purpose: "reply", Success: true, TurnID: "preview", TTFTMs: 123})

	s.eventually(func() bool { return countOf[ModelCall](s.reported()) == 1 }, "the call was never reported")
	call, _ := firstOf[ModelCall](s.reported())
	s.Equal("preview", call.TurnID, "the call is reported as the request it was")
	s.agent.turns.mu.Lock()
	defer s.agent.turns.mu.Unlock()
	s.InDelta(123, s.agent.turns.open["answer"].llmTTFTMs, 0.001)
}

func (s *AgentSuite) TestAWaitKeepsThePreviewUntilTheWordsAreChecked() {
	s.join(true)
	s.waitsThenAnswers()
	s.noRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please find a table")

	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	s.Equal(1, s.previewsHeld())
	s.Len(s.model.requests(), 1)
	s.Empty(s.voice.spoken(), "a preview is never spoken before the words are answered")
}

func (s *AgentSuite) TestAPreviewKeptForAWaitIsLetGoWhenTheWordsChange() {
	s.join(true)
	s.waitsThenAnswers()
	s.noRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	s.says(participant, "please find a table for two")

	s.eventually(kept.Load, "the preview of words that have changed was left running")
	s.Zero(s.keptPreviews())
	// Held in a local: a check that outlives the test must not read the next test's fixtures.
	agent := s.agent
	s.Never(func() bool {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		return len(agent.previews) > 1
	}, 200*time.Millisecond, 5*time.Millisecond, "two previews for one participant")
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the new words were never previewed")
	s.Equal("please find a table for two", s.model.requests()[1].Input[len(s.model.requests()[1].Input)-1].Content)
}

func (s *AgentSuite) TestAPreviewKeptForAWaitIsLetGoWhenAnotherCheckDoesNotAnswer() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.flow.then = []string{`{"disposition":"ignore","floor":"continue"}`}
	s.quickRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "please find a table")

	s.eventually(func() bool { return len(s.flow.requests()) == 2 }, "the words were never checked again")
	s.eventually(func() bool { return s.previewsHeld() == 0 && s.keptPreviews() == 0 },
		"a decision that was not an answer left the preview behind")
	s.Empty(s.voice.spoken())
	s.Zero(countOf[Responding](s.reported()))
	s.Len(s.model.requests(), 1, "the retry did not start a second preview")
}

func (s *AgentSuite) TestAPreviewKeptForAWaitIsLetGoWhenThePatienceRunsOut() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	s.agent.converse.mu.Lock()
	s.agent.converse.patience = 100 * time.Millisecond
	s.agent.converse.mu.Unlock()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	s.eventually(kept.Load, "the preview outlived the patience for its words")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestAPreviewKeptForAWaitIsLetGoWhenTheCallerLeaves() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	s.edge.attending <- Attendance{Participant: participant, Joined: false}

	s.eventually(kept.Load, "the preview of somebody who left was left running")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestClosingTheAgentLetsGoOfAPreviewKeptForAWait() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	s.Require().NoError(s.agent.Close())

	s.True(kept.Load(), "the preview was left running")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestAPreviewKeptForAWaitIsNotTakenOverByAReplyToSomethingElse() {
	// Another reply changes the conversation the kept one was written against, so it could
	// not be adopted whatever the words said.
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.says(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	_, err := s.agent.RespondTo(s.ctx, "what time is it", nil)
	s.Require().NoError(err)

	s.eventually(kept.Load, "the kept preview survived a reply that changed the conversation")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAWaitFromALowAcousticScoreKeepsThePreviewForTheNextScore() {
	var requests atomic.Int64
	s.primaryEOTServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			s.T().Errorf("read acoustic frame: %v", err)
			return
		}
		score := 0.1
		if requests.Add(1) > 1 {
			score = 0.9
		}
		writeEOTResponse(s.T(), w, r.Header.Get("X-Request-ID"), len(pcm)/2, score)
	}))
	s.join(false)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}

	s.primaryCandidate(participant, "please find a table")
	s.eventually(func() bool { return requests.Load() == 1 && s.keptPreviews() == 1 },
		"the low score did not keep the preview")
	// The score is asked again once there is something new to score.
	s.speak(participant)

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the second score did not release the caller turn")
	s.EqualValues(2, requests.Load(), "the words were scored twice")
	s.Len(s.model.requests(), 1, "the reply started before the low score is the one that was spoken")
	s.Empty(s.flow.requests(), "a valid score bypasses the semantic flow model")
	s.Zero(s.previewsHeld())
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAnInterruptionLetsGoOfAPreviewKeptForAWait() {
	s.join(true)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)
	s.says(participant, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the wait did not keep the preview")
	kept := s.watchPreview()

	s.agent.perform(Action{Kind: ActInterrupt, Participant: participant})

	s.True(kept.Load(), "the floor changed under the kept preview")
	s.Zero(s.keptPreviews())
}
