package agent

import (
	"strings"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func (s *AgentSuite) TestInterruptedReplyPrefixIsContextForTheNextCallerTurn() {
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = nil
	s.model.mu.Unlock()
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()

	firstID, err := s.agent.RespondTo(s.ctx, "What is on the menu?", nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(firstID) != nil }, "the first reply never opened")
	partial := "We have tomato soup. The mains include grilled fish and"
	s.model.writes(firstID, partial)
	s.eventually(func() bool {
		return s.currentSaying() == partial && len(s.voice.spoken()) == 1
	}, "the reply prefix was not generated and sent toward playout")

	s.agent.Interrupt()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the partial reply was not interrupted")
	s.never(func() bool { return len(s.model.requests()) > 1 },
		"an interrupted reply must not resume on its own")
	s.Equal(1, len(s.voice.spoken()), "the saved prefix must not be synthesized again")

	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal(llm.User, history[0].Role)
	s.Equal("What is on the menu?", history[0].Content)
	s.Equal(llm.Assistant, history[1].Role)
	s.Equal(partial, history[1].Content)
	s.Zero(countOf[Responded](s.reported()), "an interrupted prefix is not a completed reply")

	// A stale provider delta after the stop cannot restore the old turn's visible prefix.
	s.agent.say(firstID, "late words")
	s.Equal(partial, s.agent.History()[1].Content)
	s.Empty(s.currentSaying())

	s.model.mu.Lock()
	s.model.reply = nil
	s.model.mu.Unlock()
	_, err = s.agent.RespondTo(s.ctx, "Could you include salads?", nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return len(s.model.requests()) == 2 },
		"the new caller turn was not sent to the model")

	request := s.model.requests()[1]
	s.Require().GreaterOrEqual(len(request.Input), 3)
	tail := request.Input[len(request.Input)-3:]
	s.Equal(llm.User, tail[0].Role)
	s.Equal("What is on the menu?", tail[0].Content)
	s.Equal(llm.Assistant, tail[1].Role)
	s.Equal(partial, tail[1].Content)
	s.Equal(llm.User, tail[2].Role)
	s.Equal("Could you include salads?", tail[2].Content)
	s.Equal(1, countMessagesWithContent(request.Input, partial))
	s.Contains(request.Instructions, interruptedReplyNote)
	s.False(s.interruptionNotePending())
	for _, event := range s.reported() {
		if responded, ok := event.(Responded); ok {
			s.NotEqual(firstID, responded.TurnID)
		}
	}
}

func (s *AgentSuite) TestCompletedReplyInterruptedWhileDrainingIsNotDuplicated() {
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = []string{"The answer is seven."}
	s.model.mu.Unlock()
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()

	turnID, err := s.agent.RespondTo(s.ctx, "How many?", nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 },
		"the complete model response did not finish")
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.speakingTurn == turnID && !s.agent.generating && s.agent.utterances > 0
	}, "the completed reply did not remain in playout")

	s.agent.Interrupt()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the draining reply was not interrupted")
	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal(llm.Assistant, history[1].Role)
	s.Equal("The answer is seven.", history[1].Content)
	s.Equal(1, countMessagesWithContent(history, "The answer is seven."))
	s.Zero(countOf[Spoke](s.reported()), "the silent fixture never heard the queued audio")
	s.True(s.interruptionNotePending(), "the next turn must know the completed reply may not have been heard")
}

func (s *AgentSuite) TestInterruptWinsRaceWithACompletingReplyAndKeepsOnePrefix() {
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = nil
	s.model.mu.Unlock()
	s.voice.mu.Lock()
	s.voice.silent = true
	s.voice.mu.Unlock()

	turnID, err := s.agent.RespondTo(s.ctx, "Explain the first step.", nil)
	s.Require().NoError(err)
	s.eventually(func() bool { return s.model.script(turnID) != nil }, "the reply never opened")
	partial := "The first step is to pick a destination"
	s.model.writes(turnID, partial)
	s.eventually(func() bool { return s.currentSaying() == partial }, "the reply prefix never arrived")

	// The completed response will block in its final synthesis call while the stop wins
	// Agent.mu, drops local speech, and records the prefix. Provider cleanup is then held
	// by the same test mutex so the two terminal paths overlap deterministically.
	s.voice.mu.Lock()
	released := false
	defer func() {
		if !released {
			s.voice.mu.Unlock()
		}
	}()
	s.model.finishes(turnID)
	s.eventually(func() bool {
		s.agent.mu.Lock()
		defer s.agent.mu.Unlock()
		return s.agent.generating && s.agent.utterances > 0
	}, "completion never reached the blocked final synthesis")

	stopped := make(chan struct{})
	go func() {
		s.agent.Interrupt()
		close(stopped)
	}()
	s.eventually(func() bool {
		history := s.agent.History()
		return len(history) == 2 && history[1].Role == llm.Assistant && history[1].Content == partial
	}, "interruption did not commit the partial before provider cleanup")
	s.Zero(countOf[Responded](s.reported()), "the interrupted ResponseCompleted lost the ownership race")

	s.voice.mu.Unlock()
	released = true
	s.eventually(func() bool {
		select {
		case <-stopped:
			return true
		default:
			return false
		}
	}, "provider cleanup did not finish")
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the winning stop was not reported")
	history := s.agent.History()
	s.Require().Len(history, 2)
	s.Equal(partial, history[1].Content)
	s.Equal(1, countMessagesWithContent(history, partial))
	s.Zero(countOf[Responded](s.reported()))
	s.Empty(s.currentSaying(), "a late completion must not restore the interrupted prefix")
}

func (s *AgentSuite) TestPreviewUsesButDoesNotConsumeTheInterruptionNote() {
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = []string{"Let me finish that."}
	s.model.mu.Unlock()

	partial := "The previous answer stopped here"
	s.agent.mu.Lock()
	s.agent.history = append(s.agent.history, llm.Message{Role: llm.Assistant, Content: partial})
	s.agent.interruptedReplyPending = true
	current := s.agent.harness
	instructions := s.agent.instructions()
	s.agent.mu.Unlock()

	ready := candidate{
		ID:          replyPrefix + "continuation-test",
		Participant: stt.Participant{ID: "alice"},
		Text:        "Can you finish the list?",
		ReadyAt:     time.Now(),
		Confidence:  1,
	}
	s.agent.preview(ready, current, instructions)
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "the preview was not sent")
	s.Contains(s.model.requests()[0].Instructions, interruptedReplyNote)
	s.True(s.interruptionNotePending(), "a speculative preview must not consume the note")

	s.Require().NoError(s.agent.respondCandidate(ready, ""))
	s.eventually(func() bool { count := countOf[Responded](s.reported()); return count == 1 },
		"the accepted preview did not become the actual reply")
	s.Len(s.model.requests(), 1, "the accepted preview should avoid a duplicate model request")
	s.False(s.interruptionNotePending(), "the actual caller response consumes the note")
	request := s.model.requests()[0]
	s.Contains(request.Instructions, interruptedReplyNote)
	s.ContainsMessage(request.Input, llm.Assistant, partial)
}

func (s *AgentSuite) TestInterruptedDirectionOnlyDeltaCannotContaminateNextTurn() {
	s.performing = "You may write stage directions in square brackets."
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = nil
	s.model.mu.Unlock()

	// Give the model consumer a current owner without opening a provider stream. Direct
	// calls then place a late old completion precisely after the next delta without a
	// competing provider event or any dependency on stream timing.
	firstID := "old-direction-turn"
	s.agent.mu.Lock()
	s.agent.speakingTurn = firstID
	s.agent.generating = true
	s.agent.mu.Unlock()

	// This is an incomplete direction, so both the direction stripper and sentence
	// chunker retain it without producing visible text.
	s.agent.say(firstID, "[sigh")
	s.Equal(firstID, s.agent.replying, "direction-only output still owns the model buffers")
	s.Equal("[sigh", s.agent.chunk.pending.String())
	s.Empty(s.currentSaying(), "a stage direction is not caller-visible text")

	stopped, ok := s.agent.stopPlayback(stt.Participant{ID: "alice"}, firstID, time.Time{}, "test", "unit")
	s.Require().True(ok, "the active direction-only reply should be stoppable")
	s.agent.finishInterrupt(stopped)

	secondID := "new-direction-turn"
	s.agent.mu.Lock()
	s.agent.speakingTurn = secondID
	s.agent.generating = true
	s.agent.mu.Unlock()
	s.agent.say(secondID, "Hello.")
	s.agent.handle(llm.ResponseCompleted{Response: llm.Response{ID: firstID}})
	s.agent.finish(llm.Response{ID: secondID})

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the next reply never completed")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("Hello.", responded.Text)
	s.Equal("Hello.", said(s.voice.spoken()))
	history := s.agent.History()
	s.Equal(llm.Assistant, history[len(history)-1].Role)
	s.Equal("Hello.", history[len(history)-1].Content)
}

func (s *AgentSuite) TestCommittedDuplicateDoesNotStopAReplyBeforeItsFirstText() {
	participant, turnID, scores := s.openPrimaryReply("Tell me about dinner.", 41)

	// Flux can restate the settled utterance while the model is still opening. It is not
	// a new accepted cadence revision, so the live reply must keep its turn.
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeReplacement,
		Text:        "Tell me about dinner.",
		Utterance:   41,
		Confidence:  0.42,
	})
	s.eventually(func() bool {
		s.agent.cadence.mu.Lock()
		defer s.agent.cadence.mu.Unlock()
		return s.agent.cadence.speakers[participant.ID].confidence == 0.42
	}, "the duplicate revision was not observed")
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Text:        "Tell me about dinner.",
		Utterance:   41,
		Confidence:  0.43,
	})
	s.eventually(func() bool {
		s.agent.cadence.mu.Lock()
		defer s.agent.cadence.mu.Unlock()
		return s.agent.cadence.speakers[participant.ID].confidence == 0.43
	}, "the duplicate final was not observed after the replacement")
	s.True(s.agent.speaking(turnID), "a committed same-utterance restatement cannot abandon the opening reply")
	s.Zero(countOf[Interrupted](s.reported()))
	s.Len(s.model.requests(), 1)
	s.EqualValues(1, scores.Load(), "a duplicate must not create another EOT candidate")

	s.model.writes(turnID, "Dinner includes soup, fish, and vegetables.")
	s.model.finishes(turnID)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the preserved reply never completed")
	s.Equal(1, countOf[Responding](s.reported()))
	s.Zero(countOf[Interrupted](s.reported()), "the committed restatement cannot interrupt after the answer starts")
	responded, ok := firstOf[Responded](s.reported())
	s.Require().True(ok)
	s.Equal("Dinner includes soup, fish, and vegetables.", responded.Text)
}

func (s *AgentSuite) TestAcceptedGrowthStopsTheOpeningReplyAndAnswersTheNewCandidate() {
	initial := "Tell me about dinner."
	participant, oldTurn, scores := s.openPrimaryReply(initial, 52)
	growth := initial + " Include vegetarian options too."
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeReplacement,
		Text:        growth,
		Utterance:   52,
		Confidence:  1,
	})
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"accepted growth did not stop the stale opening reply")
	s.False(s.agent.speaking(oldTurn))
	s.Zero(countOf[Responded](s.reported()), "the held model reply has not produced a completed answer")
	s.Zero(s.overlapAsks(), "primary cadence growth bypasses semantic overlap asks")

	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Text:        growth,
		Utterance:   52,
		Confidence:  1,
	})
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 },
		"the settled grown candidate did not start its own answer")
	s.EqualValues(2, scores.Load(), "the grown candidate must receive its own primary EOT score")
	secondTurn := respondingTurn(s.reported(), 1)
	s.NotEqual(oldTurn, secondTurn, "the accepted growth must be answered as its own turn")
	s.model.writes(secondTurn, "There are lentil stew and roasted squash.")
	s.model.finishes(secondTurn)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the grown candidate answer never completed")
	s.Equal(growth, s.model.requests()[1].Input[len(s.model.requests()[1].Input)-1].Content)
}

func (s *AgentSuite) TestSameTextInANewUtteranceIsARealInterruption() {
	text := "Tell me about dinner."
	participant, oldTurn, scores := s.openPrimaryReply(text, 63)
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeReplacement,
		Text:        text,
		Utterance:   64,
		Confidence:  1,
	})
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) == 1 },
		"the same words in a new utterance should take the floor")
	s.False(s.agent.speaking(oldTurn))

	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Text:        text,
		Utterance:   64,
		Confidence:  1,
	})
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 2 },
		"a repeated utterance did not receive its own answer")
	s.EqualValues(2, scores.Load())
	secondTurn := respondingTurn(s.reported(), 1)
	s.NotEqual(oldTurn, secondTurn, "the repeated utterance must get a fresh turn")
	s.model.writes(secondTurn, "Dinner includes soup and grilled fish.")
	s.model.finishes(secondTurn)
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the repeated utterance answer never completed")
}

func (s *AgentSuite) openPrimaryReply(text string, utterance int64) (stt.Participant, string, *atomic.Int64) {
	var scores atomic.Int64
	s.primaryEOTServer(primaryScoreHandler(s, 0.9, &scores))
	s.join(true)
	s.model.mu.Lock()
	s.model.reply = nil
	s.model.mu.Unlock()
	setStubVoiceSilent(s.voice)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.speak(participant)
	s.eventually(func() bool {
		return len(s.agent.eotAudioSnapshot(participant.ID)) >= eotMinSamples*2
	}, "the caller audio window was not retained")
	s.ears.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        stt.ModeFinal,
		Text:        text,
		Utterance:   utterance,
		Language:    "en",
		Confidence:  1,
	})
	s.eventually(func() bool { return countOf[Responding](s.reported()) == 1 },
		"the initial primary-EOT candidate did not open a reply")
	turnID := respondingTurn(s.reported(), 0)
	s.eventually(func() bool { return s.model.script(turnID) != nil }, "the opening reply never reached the model")
	return participant, turnID, &scores
}

func respondingTurn(events []Event, index int) string {
	var turns []string
	for _, event := range events {
		if responding, ok := event.(Responding); ok {
			turns = append(turns, responding.TurnID)
		}
	}
	if index >= len(turns) {
		return ""
	}
	return turns[index]
}

func (s *AgentSuite) currentSaying() string {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	return s.agent.saying
}

func (s *AgentSuite) interruptionNotePending() bool {
	s.agent.mu.Lock()
	defer s.agent.mu.Unlock()
	return s.agent.interruptedReplyPending
}

func countMessagesWithContent(messages []llm.Message, content string) int {
	count := 0
	for _, message := range messages {
		if message.Content == content {
			count++
		}
	}
	return count
}

func (s *AgentSuite) ContainsMessage(messages []llm.Message, role llm.Role, content string) {
	s.T().Helper()
	for _, message := range messages {
		if message.Role == role && strings.Contains(message.Content, content) {
			return
		}
	}
	s.Failf("message not found", "want %s message containing %q", role, content)
}
