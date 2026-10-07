package agent

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// previewing has the cadence announce words that hold still for d, with the default gap so that
// the candidate for them is due long after, and the timers captured so that nothing runs until
// the test says.
func (s *CadenceSuite) previewing(d time.Duration) *[]*capturedCadenceTimer {
	s.useDefaultCadence()
	s.cadence.preview = d
	return s.captureTimers()
}

// observe is a revision of a participant's words that has not been finalized.
func (s *CadenceSuite) observe(participant stt.Participant, text string) {
	s.cadence.Observe(stt.Transcript{Participant: participant, Mode: stt.ModeReplacement, Text: text})
}

// announced is the words the cadence says a reply can be started for.
func (s *CadenceSuite) announced() candidate {
	select {
	case early := <-s.cadence.Previews():
		return early
	case <-time.After(time.Second):
		s.FailNow("no reply was announced for the words")
		return candidate{}
	}
}

// nothingAnnounced asserts no reply is announced.
func (s *CadenceSuite) nothingAnnounced() {
	select {
	case early := <-s.cadence.Previews():
		s.Failf("nothing should have been announced", "got %q", early.Text)
	case <-time.After(50 * time.Millisecond):
	}
}

func (s *CadenceSuite) TestWordsThatHoldStillForTheDebounceAreAnnouncedAheadOfTheirCandidate() {
	timers := s.previewing(150 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")

	s.Require().Len(*timers, 2)
	s.Equal(defaultCadenceGap, (*timers)[0].delay)
	s.Equal(150*time.Millisecond, (*timers)[1].delay)
	(*timers)[1].fire()
	early := s.announced()
	s.Equal(alice, early.Participant)
	s.Equal("book a table", early.Text)
	s.NotEmpty(early.ID)
	s.NotZero(early.Revision)
	s.quiet()

	(*timers)[0].fire()
	ready := s.ready()
	s.Equal(early.Revision, ready.Revision, "the candidate is for the words that were announced")
	s.NotEqual(early.ID, ready.ID)
}

func (s *CadenceSuite) TestNewWordsRestartTheDebounce() {
	timers := s.previewing(150 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	s.observe(alice, "book a table for two")

	s.Require().Len(*timers, 4)
	s.True((*timers)[1].stopped, "the debounce of the words that were replaced was left running")
	(*timers)[1].fire()
	s.nothingAnnounced()
	(*timers)[3].fire()
	early := s.announced()
	s.Equal("book a table for two", early.Text)
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestTheSameWordsAgainDoNotRestartTheDebounce() {
	timers := s.previewing(150 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}

	s.observe(alice, "book a table")
	s.observe(alice, "Book a table.")

	s.Require().Len(*timers, 2)
	s.False((*timers)[1].stopped)
}

func (s *CadenceSuite) TestWordsThatEndUnfinishedAreNotAnnounced() {
	timers := s.previewing(150 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}

	for _, text := range []string{"book a table,", "book a table and", "book a table um", "my member id is ABC12"} {
		s.observe(alice, text)
	}

	s.Require().Len(*timers, 4, "each revision is waited on, and none is announced")
	for _, timer := range *timers {
		s.Equal(defaultCadenceRetry, timer.delay)
	}
}

func (s *CadenceSuite) TestWordsWhoseCandidateIsDueNoLaterAreNotAnnounced() {
	timers := s.previewing(150 * time.Millisecond)

	s.cadence.Observe(stt.Transcript{Participant: stt.Participant{ID: "alice"}, Mode: stt.ModeFinal, Text: "book a table"})

	s.Require().Len(*timers, 1)
	s.Equal(cadenceFinalGap, (*timers)[0].delay)
}

func (s *CadenceSuite) TestWithoutADebounceNothingIsAnnounced() {
	timers := s.previewing(0)

	s.observe(stt.Participant{ID: "alice"}, "book a table")

	s.Len(*timers, 1)
}

func (s *CadenceSuite) TestACandidateForTheWordsEndsTheDebounce() {
	timers := s.previewing(150 * time.Millisecond)
	s.observe(stt.Participant{ID: "alice"}, "book a table")

	(*timers)[0].fire()
	ready := s.ready()

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
	s.Require().True(s.cadence.Resolve(ready.ID, true))
	(*timers)[1].fire()
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestForgettingAParticipantEndsTheirDebounce() {
	timers := s.previewing(150 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.observe(alice, "book a table")

	s.cadence.Forget(alice)

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
}

func (s *CadenceSuite) TestClosingEndsEveryDebounce() {
	timers := s.previewing(150 * time.Millisecond)
	s.observe(stt.Participant{ID: "alice"}, "book a table")

	s.cadence.Close()

	s.True((*timers)[1].stopped)
	(*timers)[1].fire()
	s.nothingAnnounced()
}

func TestThePreviewDebounceDefaultsToOneHundredFiftyMillisecondsAndCanBeTurnedOff(t *testing.T) {
	options := func(debounce *time.Duration) Options {
		return Options{
			CustomerID: "acme", Edge: newLoopbackEdge(), LLM: &llmrouter.Router{},
			STT: &sttrouter.Router{}, TTS: &ttsrouter.Router{}, PreviewDebounce: debounce,
		}
	}

	left, err := New(options(nil))
	require.NoError(t, err)
	require.Equal(t, 150*time.Millisecond, left.cadence.preview)

	off := time.Duration(0)
	disabled, err := New(options(&off))
	require.NoError(t, err)
	require.Zero(t, disabled.cadence.preview)

	negative := -time.Millisecond
	_, err = New(options(&negative))
	require.Error(t, err)
}

// debouncesFor has the reply to words start once they have held still for d. It is said before
// the agent joins.
func (s *AgentSuite) debouncesFor(d time.Duration) { s.previewDebounce = &d }

// slowGap leaves a revision's words unasked about for d, so that what is started for them
// ahead of the ruling can be seen before there is a ruling.
func (s *AgentSuite) slowGap(d time.Duration) {
	s.agent.cadence.mu.Lock()
	defer s.agent.cadence.mu.Unlock()
	s.agent.cadence.gap = d
}

// asked is what the conversation model was last given as the caller's words.
func (s *AgentSuite) asked(request int) string {
	input := s.model.requests()[request].Input
	return input[len(input)-1].Content
}

// holdsAnEarlyPreview joins an agent that has started a reply for a caller's words and is not
// going to ask about them, and returns the caller and what tells the reply was let go of.
func (s *AgentSuite) holdsAnEarlyPreview() (stt.Participant, *atomic.Bool) {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(time.Hour)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"no reply was started for the words")
	return alice, s.watchPreview()
}

func (s *AgentSuite) TestAStableRevisionStartsAReplyBeforeItsCandidateAndTheCandidateAdoptsIt() {
	s.debouncesFor(60 * time.Millisecond)
	s.join(true)
	s.slowGap(500 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.Equal(1, s.keptPreviews(), "it is held for the candidate")
	s.Equal(1, s.previewsHeld())
	s.Empty(s.voice.spoken(), "a preview is never spoken before the words are answered")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words were never answered")
	s.Len(s.model.requests(), 1, "the candidate took the reply over rather than asking the model again")
	s.Len(s.flow.requests(), 1)
	s.Contains(said(s.voice.spoken()), "Hello there.")
	s.Zero(s.previewsHeld())
	s.Zero(s.keptPreviews())
	s.eventually(func() bool { return countOf[Turn](s.reported()) == 1 }, "the turn was never reported")
	responding, _ := firstOf[Responding](s.reported())
	turn, _ := firstOf[Turn](s.reported())
	s.Equal(responding.TurnID, turn.TurnID)
}

func (s *AgentSuite) TestWordsThatChangeWithinTheDebounceRestartItAndOneReplyIsStartedForTheLastOnes() {
	s.debouncesFor(300 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")
	time.Sleep(150 * time.Millisecond)
	s.mutters(alice, "please find a table for two")

	s.Never(func() bool { return len(s.model.requests()) > 0 }, 250*time.Millisecond, 10*time.Millisecond,
		"a reply was started for words that changed before they held still")
	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the last words")
	s.Equal("please find a table for two", s.asked(0))
	s.Never(func() bool { return len(s.model.requests()) > 1 || s.previewsHeld() > 1 }, 400*time.Millisecond,
		10*time.Millisecond, "more than one reply for one participant")
	s.Equal(1, s.keptPreviews())
}

func (s *AgentSuite) TestWordsThatChangeAfterAReplyWasStartedLetGoOfItAndStartAnother() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 && len(s.model.requests()) == 1 },
		"no reply was started for the words")
	first := s.watchPreview()

	s.mutters(alice, "please find a table for two")

	s.eventually(first.Load, "the reply to the old words was left running")
	s.eventually(func() bool { return len(s.model.requests()) == 2 }, "no reply was started for the new words")
	s.Equal("please find a table for two", s.asked(1))
	agent := s.agent
	s.Never(func() bool {
		agent.mu.Lock()
		defer agent.mu.Unlock()
		return len(agent.previews) > 1
	}, 200*time.Millisecond, 5*time.Millisecond, "two previews for one participant")
	s.Equal(1, s.keptPreviews())
}

func (s *AgentSuite) TestWordsThatLookUnfinishedAreNotPreviewedEarly() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	for _, words := range []string{"please find a table,", "please find a table and", "please find a table um",
		"my member id is ABC12"} {
		s.mutters(alice, words)
		s.Never(func() bool { return len(s.model.requests()) > 0 }, 150*time.Millisecond, 10*time.Millisecond,
			"a reply was started for words that end unfinished: "+words)
	}

	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestWithoutADebounceTheReplyStartsWithTheCandidate() {
	s.debouncesFor(0)
	s.join(true)
	s.slowGap(400 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 }, 250*time.Millisecond, 10*time.Millisecond,
		"a reply was started ahead of the candidate")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the words were never answered")
	s.Len(s.model.requests(), 1)
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWithoutSpeculativeReplies() {
	off := false
	s.speculation = &off
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words nobody asked about")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWhileTheAgentIsSpeaking() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	s.agent.mu.Lock()
	s.agent.utterances = 1
	s.agent.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words said over the agent")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyWhenAPolicyMustClearTheWordsFirst() {
	s.screens(guardrail.ModeBlocking)
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.Never(func() bool { return len(s.model.requests()) > 0 || s.previewsHeld() > 0 },
		250*time.Millisecond, 10*time.Millisecond, "a reply was started for words the policy had not cleared")
}

func (s *AgentSuite) TestNoReplyIsStartedEarlyForAnotherVoiceAtTheMicrophone() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(10 * time.Second)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Speaker: "voice-1", Text: "please find a table"})
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "the caller's own voice was not previewed")
	started := len(s.model.requests())

	s.ears.emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Speaker: "voice-2", Text: "who is at the door"})

	s.Never(func() bool { return len(s.model.requests()) > started }, 250*time.Millisecond, 10*time.Millisecond,
		"a reply was started for somebody else's words")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenTheFloorChanges() {
	alice, started := s.holdsAnEarlyPreview()

	s.agent.perform(Action{Kind: ActInterrupt, Participant: alice})

	s.True(started.Load(), "the floor changed under the reply")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenTheCallerLeaves() {
	alice, started := s.holdsAnEarlyPreview()

	s.edge.attending <- Attendance{Participant: alice, Joined: false}

	s.eventually(started.Load, "the reply for somebody who left was left running")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestClosingTheAgentLetsGoOfAnEarlyPreview() {
	_, started := s.holdsAnEarlyPreview()

	s.Require().NoError(s.agent.Close())

	s.True(started.Load(), "the reply was left running")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestMovingASessionOntoOtherModelsLetsGoOfAnEarlyPreview() {
	w := s.joinSwappable()
	s.agent.cadence.mu.Lock()
	s.agent.cadence.preview = 50 * time.Millisecond
	s.agent.cadence.gap = time.Hour
	s.agent.cadence.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.eventually(func() bool {
		s.agent.mu.Lock()
		_, listening := s.agent.listeners[alice.ID]
		s.agent.mu.Unlock()
		return listening && len(w.listener().transcribed()) > 0
	}, "nobody listened")
	w.listener().emitter.Send(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "please find a table", Language: "en"})
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "no reply was started for the words")
	started := s.watchPreview()

	s.Require().NoError(s.agent.SetSettings(s.ctx, Settings{
		LLMTarget: "other/other-model", STTTarget: "stub/stub-model", TTSTarget: "other/other-model", Voice: "ada",
	}))

	s.True(started.Load(), "the reply was written by the model that was replaced")
	s.Zero(s.keptPreviews())
}

func (s *AgentSuite) TestARulingThatIsNotAnAnswerLetsGoOfAnEarlyPreview() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(300 * time.Millisecond)
	s.flow.reply = []string{`{"disposition":"ignore","floor":"continue"}`}
	alice := stt.Participant{ID: "child", Name: "Child"}
	s.speak(alice)

	s.mutters(alice, "mom where is my backpack")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the words were never asked about")
	s.eventually(func() bool { return s.previewsHeld() == 0 && s.keptPreviews() == 0 },
		"words that were ignored kept the reply started for them")
	s.Len(s.model.requests(), 1, "the candidate did not start a second reply")
	s.Zero(countOf[Responding](s.reported()))
	s.Empty(s.voice.spoken())
}

func (s *AgentSuite) TestAnEarlyPreviewIsLetGoWhenNothingComesOfTheWordsBeforeThePatienceRunsOut() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(time.Hour)
	s.agent.converse.mu.Lock()
	s.agent.converse.patience = 150 * time.Millisecond
	s.agent.converse.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return s.keptPreviews() == 1 }, "no reply was started for the words")
	started := s.watchPreview()

	s.eventually(started.Load, "the reply outlived the patience for its words")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
}

func (s *AgentSuite) TestAnEarlyPreviewAdoptedByAWaitIsLetGoWhenThePatienceRunsOut() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.slowGap(200 * time.Millisecond)
	s.flow.reply = []string{`{"disposition":"wait","floor":"continue"}`}
	s.noRetry()
	s.agent.converse.mu.Lock()
	s.agent.converse.patience = 300 * time.Millisecond
	s.agent.converse.mu.Unlock()
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)
	s.mutters(alice, "please find a table")
	s.eventually(func() bool { return len(s.flow.requests()) == 1 }, "the words were never asked about")
	s.eventually(func() bool { return s.keptPreviews() == 1 && s.previewsHeld() == 1 }, "the wait did not keep the reply")
	started := s.watchPreview()

	s.eventually(started.Load, "the reply outlived the patience for its words")
	s.Zero(s.keptPreviews())
	s.Zero(s.previewsHeld())
	s.Len(s.model.requests(), 1, "the wait kept the reply that was started early")
}

func (s *AgentSuite) TestAnEarlyPreviewThatFailedLeavesNothingHeld() {
	s.debouncesFor(50 * time.Millisecond)
	s.join(true)
	s.model.refuses = errors.New("the model is down")
	s.slowGap(300 * time.Millisecond)
	alice := stt.Participant{ID: "alice"}
	s.speak(alice)

	s.mutters(alice, "please find a table")

	s.eventually(func() bool { return len(s.model.requests()) == 1 }, "no reply was started for the words")
	s.Empty(s.flow.requests(), "the reply was started before the words were put to a ruling")
	s.eventually(func() bool { return countOf[Error](s.reported()) > 0 }, "the failure was never reported")
	s.eventually(func() bool { return s.previewsHeld() == 0 && s.keptPreviews() == 0 },
		"a reply that failed was left held")
	s.Empty(s.voice.spoken())
}
