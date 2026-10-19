package agent

import (
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// The cadence is driven at a fraction of its real gap, because what these assert is what
// is decided rather than how long the deciding waits.
const (
	testGap    = 10 * time.Millisecond
	testRetry  = 20 * time.Millisecond
	testWithin = time.Second
	// testPatience is shorter than one retry, so the second ruling about words nobody has
	// added to is the one that gives up on waiting for them.
	testPatience = 5 * time.Millisecond
)

var caller = stt.Participant{ID: "caller", UserID: "caller", Name: "Alex"}

type ConverseSuite struct {
	suite.Suite

	settling *cadence
	emitter  *Emitter
	converse *converse

	drained sync.WaitGroup
	mu      sync.Mutex
	events  []Event
}

func TestConverseSuite(t *testing.T) {
	suite.Run(t, new(ConverseSuite))
}

func (s *ConverseSuite) SetupTest() {
	s.build(DuplexOptions{})
}

// build starts a conversation with the given listening options and watches everything it
// reports, which is how the decision trail is asserted on.
func (s *ConverseSuite) build(options DuplexOptions) {
	s.start(options, testPatience)
}

// start is build, for the tests that also care how long an unfinished thought is waited on.
func (s *ConverseSuite) start(options DuplexOptions, patience time.Duration) {
	s.startWithSettle(options, patience, time.Hour)
}

func (s *ConverseSuite) startWithSettle(options DuplexOptions, patience, settle time.Duration) {
	s.teardown()

	logger := slog.New(slog.DiscardHandler)
	s.settling = newCadence(testGap, testRetry, settle, logger)
	s.emitter = NewEmitter(eventBuffer)
	s.converse = newConverse(s.settling, newDuplex(options), s.emitter, nil, patience, logger)

	s.mu.Lock()
	s.events = nil
	s.mu.Unlock()

	s.drained.Add(1)
	go func() {
		defer s.drained.Done()
		for event := range s.emitter.Events() {
			s.mu.Lock()
			s.events = append(s.events, event)
			s.mu.Unlock()
		}
	}()
}

func (s *ConverseSuite) TearDownTest() { s.teardown() }

func (s *ConverseSuite) teardown() {
	if s.emitter == nil {
		return
	}
	s.settling.Close()
	s.emitter.Close()
	s.drained.Wait()
	s.emitter = nil
}

// settle says something and waits for the words to hold still, which is the state every
// ruling is made against. The words are observed as a quiet floor so a test of a settled
// ruling does not also occupy the controller with a provisional overlap ask; tests that
// want that ask use overhears.
func (s *ConverseSuite) settle(text string, state floor) candidate {
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        text,
	}, s.quiet())

	return s.converse.Settled(s.held(), state).Candidate
}

// held is the next turn whose words have stopped changing, whether they are being put for
// the first time or put again after the conversation decided to wait on them.
func (s *ConverseSuite) held() candidate {
	select {
	case ready := <-s.settling.Ready():
		return ready
	case <-time.After(testWithin):
		s.FailNow("the words never held still")
		return candidate{}
	}
}

func (s *ConverseSuite) ignoreNextSettled() {
	s.T().Helper()
	ready := s.held()
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Ignore,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActIgnore}, kinds(actions))
}

// talking is an agent mid-reply, which is what makes the floor half of a ruling matter.
func (s *ConverseSuite) talking() floor {
	return floor{Speaking: "turn-1", LastParticipant: caller}
}

func (s *ConverseSuite) quiet() floor {
	return floor{Quiet: true, LastParticipant: caller}
}

func (s *ConverseSuite) seen() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Event(nil), s.events...)
}

// eventually gives the emitter a moment to hand an event to the watcher, since sending is
// asynchronous and an assertion made straight away would race it.
func (s *ConverseSuite) eventually(satisfied func() bool, message string) {
	s.T().Helper()
	deadline := time.Now().Add(testWithin)
	for time.Now().Before(deadline) {
		if satisfied() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	s.FailNow(message)
}

func (s *ConverseSuite) heard() []Heard {
	var kept []Heard
	for _, event := range s.seen() {
		if typed, ok := event.(Heard); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

// overheard plays an in-progress revision over the agent and returns the provisional asks
// it produced.
func (s *ConverseSuite) overheard(text string, state floor) []candidate {
	var asks []candidate
	for _, action := range s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: text,
	}, state) {
		if action.Kind == ActAsk {
			asks = append(asks, action.Candidate)
		}
	}
	return asks
}

// overhears is overheard for a revision that must produce exactly one ask.
func (s *ConverseSuite) overhears(text string, state floor) candidate {
	asks := s.overheard(text, state)
	s.Require().Len(asks, 1, "no provisional ask was made about the overlap")
	return asks[0]
}

// queueOnOverlap starts a numbered transcript while the agent is speaking, then lets its
// settled candidate meet the still-in-flight overlap ask. This is the path where the words
// are queued before a provisional ruling comes back.
func (s *ConverseSuite) queueOnOverlap(transcript stt.Transcript) (candidate, candidate) {
	var provisional candidate
	for _, action := range s.converse.Observe(transcript, s.talking()) {
		if action.Kind == ActAsk {
			provisional = action.Candidate
		}
	}
	s.Require().NotEmpty(provisional.ID, "the overlapping words were not asked about")

	ready := s.held()
	action := s.converse.Settled(ready, s.talking())
	s.Require().Equal(ActQueue, action.Kind, "settled words wait behind their overlap ask")
	return ready, provisional
}

func (s *ConverseSuite) stopOverlapAsk(provisional candidate) {
	s.T().Helper()
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: provisional.ID,
		Disposition: harness.Respond,
		Floor:       harness.Stop,
	}, s.talking())
	s.Equal([]ActionKind{ActInterrupt}, kinds(actions))
}

func (s *ConverseSuite) queueSettledTurn(text string, utterance int64, floorDecision harness.Floor) candidate {
	s.T().Helper()
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        text,
	}, s.quiet())
	ready := s.held()
	asked := s.converse.Settled(ready, s.talking())
	s.Require().Equal(ActAsk, asked.Kind)

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       floorDecision,
	}, s.talking())
	if floorDecision == harness.Shorten {
		s.Equal([]ActionKind{ActQueue, ActShorten}, kinds(actions))
	} else {
		s.Equal([]ActionKind{ActQueue}, kinds(actions))
	}
	return ready
}

func (s *ConverseSuite) answerFinal(text string, utterance int64) Action {
	s.T().Helper()
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
		Text:        text,
	}, s.quiet())
	ready := s.held()
	s.Equal(text, ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActAnswer}, kinds(actions))
	s.Equal(text, actions[0].Text)
	return actions[0]
}

func (s *ConverseSuite) assertCorrectedReplacedQueue(corrected string) {
	s.T().Helper()
	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "answering the correction must consume its older queued hypothesis")
	s.eventually(func() bool {
		heard := s.heard()
		return len(heard) == 1 && heard[0].Text == corrected
	}, "the corrected utterance was not reported exactly once")
	for _, heard := range s.heard() {
		s.NotEqual("Way", heard.Text, "the obsolete hypothesis must never be replayed")
	}

	// A repeated final is the provider settling the accepted utterance, not a second turn.
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   7,
		Text:        corrected,
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("a duplicate final became another turn", "got %q", duplicate.Text)
	case <-time.After(interruptGrace + testGap + 3*testRetry):
	}
}

func (s *ConverseSuite) holds() int {
	var kept int
	for _, decision := range s.decisions() {
		if decision.Kind == string(ActHold) {
			kept++
		}
	}
	return kept
}

func (s *ConverseSuite) TestWhatAProvisionalRulingDoesAboutTheFloor() {
	// A ruling made while the caller is still talking decides only the floor: it never
	// answers, because the words have not settled.
	cases := []struct {
		name        string
		disposition harness.Disposition
		floor       harness.Floor
		expected    []ActionKind
	}{
		{"a correction takes the floor", harness.Wait, harness.Stop, []ActionKind{ActInterrupt}},
		{"a related addition shortens the reply", harness.Wait, harness.Shorten, []ActionKind{ActShorten}},
		{"an acknowledgement lets it keep talking", harness.Wait, harness.Continue, []ActionKind{}},
		{"background speech does not stop it", harness.Ignore, harness.Stop, []ActionKind{}},
	}

	for _, test := range cases {
		s.Run(test.name, func() {
			s.build(DuplexOptions{})
			asked := s.overhears("actually wait", s.talking())

			actions := s.converse.Ruled(harness.Decided{
				CandidateID: asked.ID,
				Disposition: test.disposition,
				Floor:       test.floor,
			}, s.talking())

			s.Equal(test.expected, kinds(actions))
			for _, action := range actions {
				if action.Kind == ActInterrupt || action.Kind == ActShorten {
					s.Equal("turn-1", action.TurnID, "the reply in flight is what is cut")
				}
			}
		})
	}
}

func (s *ConverseSuite) TestAProvisionalRulingThatArrivedTooLateIsDropped() {
	stop := func(id string, state floor) []Action {
		return s.converse.Ruled(harness.Decided{
			CandidateID: id, Disposition: harness.Wait, Floor: harness.Stop,
		}, state)
	}

	s.Run("the floor went quiet", func() {
		s.build(DuplexOptions{})
		asked := s.overhears("actually wait", s.talking())
		s.Empty(kinds(stop(asked.ID, s.quiet())))
	})

	s.Run("the reply moved on", func() {
		s.build(DuplexOptions{})
		asked := s.overhears("actually wait", s.talking())
		s.Empty(kinds(stop(asked.ID, floor{Speaking: "turn-2", LastParticipant: caller})))
	})

	s.Run("a new utterance replaced the words", func() {
		s.build(DuplexOptions{})
		asked := s.overhears("actually wait", s.talking())
		s.converse.Observe(stt.Transcript{
			Participant: caller, Mode: stt.ModeReplacement, Text: "something else entirely",
		}, s.talking())
		s.Empty(kinds(stop(asked.ID, s.talking())))
	})
}

func (s *ConverseSuite) TestHowOftenAnOverlapIsAskedAbout() {
	s.build(DuplexOptions{})

	// Each ask is ruled continue so the next revision is not blocked by one in flight.
	ask := func(text string) int {
		asks := s.overheard(text, s.talking())
		for _, asked := range asks {
			s.converse.Ruled(harness.Decided{
				CandidateID: asked.ID, Disposition: harness.Wait, Floor: harness.Continue,
			}, s.talking())
		}
		return len(asks)
	}

	s.Equal(0, ask("one"), "a lone word that is no interruption waits for another")
	s.Equal(1, ask("one two"), "the first revision with two words is asked about")
	s.Equal(0, ask("one two three"), "one more word is not enough to ask again")
	s.Equal(1, ask("one two three four five"), "three more words asks again")
	s.Equal(1, ask("one two three four five six seven eight"), "and again")
	s.Equal(0, ask("one two three four five six seven eight nine ten eleven"),
		"a fourth ask on the same utterance is not made")
	s.Equal(1, ask("something completely different now"),
		"a new utterance starts the count again")
}

func (s *ConverseSuite) TestALoneWordOverTheAgentIsAskedAboutOnlyWhenItInterrupts() {
	// Voicebench's child said "Mom" over the agent, Gemma ruled it a stop, and the reply was
	// abandoned for words meant for somebody else.
	for text, asks := range map[string]int{"Mom": 0, "okay": 0, "wait": 1, "No,": 1, "Hang": 1, "Mom, can": 1} {
		s.build(DuplexOptions{})
		s.Equal(asks, len(s.overheard(text, s.talking())), text)
	}
}

func (s *ConverseSuite) TestSomeThingsOverTheAgentAreNeverAskedAbout() {
	s.Run("a cough", func() {
		s.build(DuplexOptions{})
		s.Zero(len(s.overheard("cough", s.talking())), "a cough is not a reason to stop")
	})

	s.Run("speech over a murmur", func() {
		s.build(DuplexOptions{})
		s.Zero(len(s.overheard("go on then", floor{Speaking: "back-1", LastParticipant: caller})),
			"a murmur is meant to overlap, so there is nothing to abandon")
	})
}

func (s *ConverseSuite) TestAGrowingOverlapReplacesTheAskInFlight() {
	s.build(DuplexOptions{})
	first := s.overhears("okay so", s.talking())

	var superseded, asked Action
	for _, action := range s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "okay so wait make it six",
	}, s.talking()) {
		switch action.Kind {
		case ActSupersede:
			superseded = action
		case ActAsk:
			asked = action
		}
	}

	s.Equal(first.ID, superseded.TurnID, "the first ask is about words that have changed")
	s.NotEqual(first.ID, asked.Candidate.ID)
	s.Equal("okay so wait make it six", asked.Text)
	s.True(asked.Candidate.Unfinished)

	s.Empty(kinds(s.converse.Ruled(harness.Decided{
		CandidateID: first.ID, Disposition: harness.Wait, Floor: harness.Continue,
	}, s.talking())), "the superseded ruling is dropped")

	stopped := s.converse.Ruled(harness.Decided{
		CandidateID: asked.Candidate.ID, Disposition: harness.Wait, Floor: harness.Stop,
	}, s.talking())
	s.Equal([]ActionKind{ActInterrupt}, kinds(stopped),
		"the later words take the floor while the agent is still talking")
}

func (s *ConverseSuite) TestASettledTurnWaitsOnAnOverlapAskInFlight() {
	s.build(DuplexOptions{})
	s.overhears("okay wait make it six", s.talking())

	ready := s.held()
	action := s.converse.Settled(ready, s.talking())
	s.Equal(ActQueue, action.Kind,
		"retrying cadence would re-Settle every gap until the overlap Create returned")
	s.Equal(ready.ID, action.Candidate.ID)

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(ActAnswer, action.Kind)
	s.Equal(ready.ID, action.Candidate.ID)
}

func (s *ConverseSuite) TestATurnQueuedBehindAnOverlapAskIsHeardOnceItIsAnswered() {
	// A transcript closes the caller's line on Heard. A turn that never sends it leaves the
	// line open, and whatever the caller says next is written over it.
	s.build(DuplexOptions{})
	s.overhears("okay wait make it six", s.talking())
	queued := s.converse.Settled(s.held(), s.talking())
	s.Require().Equal(ActQueue, queued.Kind)

	_, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.eventually(func() bool {
		for _, heard := range s.heard() {
			if heard.Text == "okay wait make it six" {
				return true
			}
		}
		return false
	}, "the queued turn was answered without being reported as what the caller said")
}

func (s *ConverseSuite) TestATurnQueuedAfterItsRulingIsHeardOnlyOnce() {
	s.build(DuplexOptions{})
	ready := s.settle("book a table for four", s.talking())
	queued := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.talking())
	s.Require().Equal([]ActionKind{ActQueue}, kinds(queued))
	_, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)

	// Events arrive in order, so once a later turn is heard, a second report of the queued
	// one would already be here.
	later := s.settle("and a high chair", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: later.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())
	s.eventually(func() bool {
		heard := s.heard()
		return len(heard) > 0 && heard[len(heard)-1].Text == "and a high chair"
	}, "the later turn was never heard")

	var reports int
	for _, heard := range s.heard() {
		if heard.Text == "book a table for four" {
			reports++
		}
	}
	s.Equal(1, reports, "a turn ruled on before it was queued was reported again when answered")
}

func (s *ConverseSuite) TestAQueuedTurnIsDroppedOnceTheSameWordsAreAnsweredInFull() {
	// The caller was still talking when their first words settled, so those were queued.
	// Everything they went on to say is answered, and the queued part must not be answered
	// again, late, once the agent stops.
	s.build(DuplexOptions{})
	s.overhears("okay wait make it six", s.talking())
	queued := s.converse.Settled(s.held(), s.talking())
	s.Require().Equal(ActQueue, queued.Kind)

	ready := s.settle("okay wait make it six people please", s.quiet())
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the queued words were answered as part of the longer turn")
}

func (s *ConverseSuite) TestAReplacementThenFinalCorrectionReplacesAQueuedOverlap() {
	const utterance = 7
	const corrected = "Wait. Make it six, and one of us is gluten free instead."

	s.build(DuplexOptions{})
	_, provisional := s.queueOnOverlap(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        "Way",
	})
	s.stopOverlapAsk(provisional)
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        corrected,
	}, s.quiet())

	answered := s.answerFinal(corrected, utterance)
	s.Equal(corrected, answered.Text)
	s.assertCorrectedReplacedQueue(corrected)
}

func (s *ConverseSuite) TestTwoReplacementsThenFinalCorrectionReplacesAQueuedOverlap() {
	const utterance = 7
	const corrected = "Wait. Make it six, and one of us is gluten free instead."

	s.build(DuplexOptions{})
	_, provisional := s.queueOnOverlap(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        "Way",
	})
	s.stopOverlapAsk(provisional)
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        "Wait.",
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        corrected,
	}, s.quiet())

	answered := s.answerFinal(corrected, utterance)
	s.Equal(corrected, answered.Text)
	s.assertCorrectedReplacedQueue(corrected)
}

func (s *ConverseSuite) TestAFinalCorrectionOfAQueuedUtteranceIsNotSwallowedAsTheAnsweredCopy() {
	const utterance = 7
	const corrected = "Wait. Make it six, and one of us is gluten free instead."

	_, provisional := s.queueOnOverlap(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        "Way",
	})
	s.stopOverlapAsk(provisional)

	// Some providers correct a hypothesis only when they finalize it. This is a new reading
	// of the same numbered utterance, not the provider replaying the already queued words.
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
		Text:        corrected,
	}, s.quiet())
	ready := s.held()
	s.Equal(corrected, ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActAnswer}, kinds(answered))
	s.Equal(corrected, answered[0].Text)

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the corrected final replaced the queued hypothesis")
}

func (s *ConverseSuite) TestARevisedContinueQueueIsNotReplayed() {
	s.assertRevisedQueueIsNotReplayed(harness.Continue)
}

func (s *ConverseSuite) TestARevisedShortenQueueIsNotReplayed() {
	s.assertRevisedQueueIsNotReplayed(harness.Shorten)
}

func (s *ConverseSuite) assertRevisedQueueIsNotReplayed(decision harness.Floor) {
	s.T().Helper()
	const utterance = 7
	const corrected = "Wait. Make it six, and one of us is gluten free instead."
	s.queueSettledTurn("Way", utterance, decision)
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        corrected,
	}, s.quiet())

	s.answerFinal(corrected, utterance)
	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the accepted correction replaces a queued Continue or Shorten turn")
}

func (s *ConverseSuite) TestAQueuedTurnSurvivesASeparateNumberedUtteranceWithTheSamePrefix() {
	const queuedText = "Way"
	s.queueSettledTurn(queuedText, 7, harness.Continue)

	const nextText = "Way make it six"
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        nextText,
	}, s.quiet())
	ready := s.held()
	s.Equal(nextText, ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting, "a new utterance is not a correction of the queued one")
	s.Equal(queuedText, action.Candidate.Text)
}

func (s *ConverseSuite) TestAQueuedTurnKeepsItsParticipantWhenUtteranceNumbersMatch() {
	const queuedText = "Way"
	s.queueSettledTurn(queuedText, 7, harness.Continue)

	other := stt.Participant{ID: "other", UserID: "other", Name: "Jordan"}
	const nextText = "Way make it six"
	s.converse.Observe(stt.Transcript{
		Participant: other,
		Mode:        stt.ModeFinal,
		Utterance:   7,
		Text:        nextText,
	}, s.quiet())
	ready := s.held()
	s.Equal(other, ready.Participant)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting, "another participant's utterance is not a correction")
	s.Equal(caller, action.Candidate.Participant)
	s.Equal(queuedText, action.Candidate.Text)
}

func (s *ConverseSuite) TestCorrectingAnOlderQueuedUtterancePreservesTheActiveNumberedTurn() {
	const corrected = "Wait. Make it six, and one of us is gluten free instead."
	s.queueSettledTurn("Way", 7, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        "No high chair.",
	}, s.quiet())
	active := s.held()
	s.Equal("No high chair.", active.Text)
	s.Require().Equal(ActAsk, s.converse.Settled(active, s.quiet()).Kind)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   7,
		Text:        corrected,
	}, s.quiet())
	ready := s.held()
	s.Equal("No high chair. "+corrected, ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActAnswer}, kinds(answered))

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the revised older utterance replaces its queued copy")
}

func (s *ConverseSuite) TestACorrectingAQueuedUtteranceKeepsItsCarriedPrefix() {
	const carried = "Last name Alvarez."
	const firstCorrection = "Last name Alvarez. Way"
	const corrected = "Last name Alvarez. Wait. Make it six, and one of us is gluten free instead."
	const tail = "Wait. Make it six, and one of us is gluten free instead."
	const utterance = 7

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   6,
		Text:        carried,
	}, s.quiet())
	first := s.held() // Drain, but do not settle or answer, the still-unanswered first utterance.
	s.Equal(carried, first.Text)

	var provisional candidate
	for _, action := range s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        "Way",
	}, s.talking()) {
		if action.Kind == ActAsk {
			provisional = action.Candidate
		}
	}
	s.Require().NotEmpty(provisional.ID)
	ready := s.held()
	s.Equal(firstCorrection, ready.Text, "the new utterance carries forward words the caller was still owed")
	queued := s.converse.Settled(ready, s.talking())
	s.Require().Equal(ActQueue, queued.Kind)
	s.stopOverlapAsk(provisional)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        tail,
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
		Text:        tail,
	}, s.quiet())
	correctedReady := s.held()
	s.Equal(corrected, correctedReady.Text,
		"correcting the tail keeps the carried prefix once and replaces only the old tail")
	asked := s.converse.Settled(correctedReady, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: correctedReady.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActAnswer}, kinds(answered))
	s.Equal(corrected, answered[0].Text)

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the full corrected turn replaced the queued partial transcript")
}

func (s *ConverseSuite) TestAZeroUtteranceIDUsesLexicalGrowthToReplaceItsQueuedCopy() {
	const queuedText = "book a table"
	const corrected = "book a table for four"
	s.queueSettledTurn(queuedText, 0, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        corrected,
	}, s.quiet())
	answered := s.answerFinal(corrected, 0)
	s.Equal(corrected, answered.Text)

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "without an utterance number, a lexical growth still replaces its queued prefix")
}

func (s *ConverseSuite) TestAnsweringAnOlderQueueDoesNotMakeANewerAnsweredUtteranceNewAgain() {
	const laterText = "what time do you close"
	s.queueSettledTurn("book a table", 7, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        laterText,
	}, s.quiet())
	later := s.held()
	asked := s.converse.Settled(later, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: later.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))

	queued, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal("book a table", queued.Candidate.Text)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        laterText,
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("answering an older queued turn replaced the newer deduplication record", "got %q", duplicate.Text)
	case <-time.After(interruptGrace + testGap + 3*testRetry):
	}
}

func (s *ConverseSuite) TestAnUnchangedNumberedQueueWaitsOnceWithoutStartingAnotherCadence() {
	const utterance = 7
	const text = "book a table"
	s.queueSettledTurn(text, utterance, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
		Text:        text,
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
		Text:        text,
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("an unchanged held transcript started another cadence", "got %q", duplicate.Text)
	case <-time.After(interruptGrace + testGap + 3*testRetry):
	}

	queued, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(ActAnswer, queued.Kind)
	s.Equal(text, queued.Candidate.Text)
	_, waiting = s.converse.Waiting(s.quiet())
	s.False(waiting, "the queued utterance is handed back exactly once")

	const laterText = "what time do you close"
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        laterText,
	}, s.quiet())
	later := s.held()
	asked := s.converse.Settled(later, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	s.converse.Ruled(harness.Decided{
		CandidateID: later.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.eventually(func() bool {
		heard := s.heard()
		return len(heard) >= 2 && heard[len(heard)-1].Text == laterText
	}, "the later transcript was not heard after the queued turn")
	var originalReports int
	for _, heard := range s.heard() {
		if heard.Text == text {
			originalReports++
		}
	}
	s.Equal(1, originalReports, "the queued turn was reported once before and after Waiting")
}

func (s *ConverseSuite) TestAnUnchangedZeroIDQueueWaitsOnceWithoutStartingAnotherCadence() {
	const text = "book a table"
	s.queueSettledTurn(text, 0, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        text,
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Text:        text,
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("an unchanged zero-ID held transcript started another cadence", "got %q", duplicate.Text)
	case <-time.After(interruptGrace + testGap + 3*testRetry):
	}

	queued, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(text, queued.Candidate.Text)
	_, waiting = s.converse.Waiting(s.quiet())
	s.False(waiting, "the zero-ID queue is handed back exactly once")
}

func (s *ConverseSuite) TestARepeatedZeroIDQueueCanStartANewTurnAfterTheSettleWindow() {
	const text = "book a table"
	const settle = 100 * time.Millisecond
	s.startWithSettle(DuplexOptions{}, testPatience, settle)
	s.queueSettledTurn(text, 0, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        text,
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("an immediate zero-ID restatement started another cadence", "got %q", duplicate.Text)
	case <-time.After(settle / 2):
	}

	time.Sleep(settle + testGap)
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        text,
	}, s.quiet())
	ready := s.held()
	s.Equal(text, ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Require().Equal([]ActionKind{ActAnswer}, kinds(answered))

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "a repeated zero-ID turn replaces the old queued copy")
}

func (s *ConverseSuite) TestBlankNumberedRevisionsKeepTheQueuedCandidate() {
	const utterance = 7
	const text = "book a table"
	s.queueSettledTurn(text, utterance, harness.Continue)

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Utterance:   utterance,
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeDelta,
		Utterance:   utterance,
		Text:        " ",
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
	}, s.quiet())

	queued, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(text, queued.Candidate.Text)
	_, waiting = s.converse.Waiting(s.quiet())
	s.False(waiting, "blank revisions do not release or duplicate the queued candidate")
}

func (s *ConverseSuite) TestAnAnsweredNumberedQueueDeduplicatesItsFinalButNotTheNextUtterance() {
	s.queueSettledTurn("1 2 3", 7, harness.Continue)
	queued, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal("1 2 3", queued.Candidate.Text)

	// Gemini settles an answered number in words after streaming it as digits.
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   7,
		Text:        "one two three",
	}, s.quiet())
	select {
	case duplicate := <-s.settling.Ready():
		s.Failf("a final respelling of the answered utterance became another turn", "got %q", duplicate.Text)
	case <-time.After(interruptGrace + testGap + 3*testRetry):
	}

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   8,
		Text:        "one two three",
	}, s.quiet())
	ready := s.held()
	s.Equal("one two three", ready.Text)
	asked := s.converse.Settled(ready, s.quiet())
	s.Require().Equal(ActAsk, asked.Kind)
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))
}

func (s *ConverseSuite) TestAQueuedTurnSurvivesDifferentWordsBeingAnswered() {
	// A transcriber that sends only what is new would make the later turn the only copy of
	// its own words, not of the queued ones, so those are still owed.
	s.build(DuplexOptions{})
	s.overhears("okay wait make it six", s.talking())
	queued := s.converse.Settled(s.held(), s.talking())
	s.Require().Equal(ActQueue, queued.Kind)

	ready := s.settle("what time do you close", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(queued.Candidate.ID, action.Candidate.ID)
}

func (s *ConverseSuite) TestAQueuedTurnTheCallerKeepsAddingToIsNotAnsweredOnItsOwn() {
	// What a caller's long turn looked like in Voicebench: they paused after "the name for the
	// booking is" while the agent talked, so that much was queued, then went on. Answering
	// the queued half once the agent stopped cut into the rest of the sentence.
	s.build(DuplexOptions{})
	s.overhears("the name for the booking is", s.talking())
	queued := s.converse.Settled(s.held(), s.talking())
	s.Require().Equal(ActQueue, queued.Kind)

	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "the name for the booking is Alvarez",
	}, s.quiet())
	s.ignoreNextSettled()

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the half sentence is answered with the rest, once that settles")
}

func (s *ConverseSuite) TestAQueuedTurnSurvivesANewTurnThatStartsTheSameWay() {
	s.build(DuplexOptions{})
	s.overhears("yes that is right", s.talking())
	queued := s.converse.Settled(s.held(), s.talking())
	s.Require().Equal(ActQueue, queued.Kind)

	s.converse.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "yes"}, s.quiet())
	s.ignoreNextSettled()

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting, "words that only start like the queued turn do not replace it")
	s.Equal(queued.Candidate.ID, action.Candidate.ID)
}

func (s *ConverseSuite) TestAnAnnouncedMonologueIsNotAnsweredUntilItEnds() {
	announced := s.settle("I'll give you everything up front so we don't go back and forth.", s.quiet())
	held := s.converse.Ruled(harness.Decided{
		CandidateID: announced.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActWait}, kinds(held))

	detail := s.settle("I'll give you everything up front so we don't go back and forth. Member ID ABC123456.", s.quiet())
	held = s.converse.Ruled(harness.Decided{
		CandidateID: detail.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActWait}, kinds(held), "a complete detail is not the end of what was announced")

	done := s.settle("I'll give you everything up front so we don't go back and forth. Member ID ABC123456. "+
		"That's everything.", s.quiet())
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: done.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(answered))
}

func (s *ConverseSuite) TestAfterAProvisionalStopTheSettledWordsAreAnsweredOnce() {
	asked := s.overhears("actually wait", s.talking())
	stopped := s.converse.Ruled(harness.Decided{
		CandidateID: asked.ID, Disposition: harness.Wait, Floor: harness.Stop,
	}, s.talking())
	s.Require().Equal([]ActionKind{ActInterrupt}, kinds(stopped))

	ready := s.settle("actually wait, make it six", s.quiet())
	answered := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.quiet())

	s.Equal([]ActionKind{ActAnswer}, kinds(answered),
		"the settled words are answered with no second interruption")
}

func (s *ConverseSuite) TestAQueuedTurnWaitsWhileTheCallerIsStillTalking() {
	// A turn queued behind a reply is owed to the caller, but not while they are talking:
	// answering then would be the agent taking the floor from somebody who has it.
	ready := s.settle("and make it eight", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID, Disposition: harness.Respond, Floor: harness.Continue,
	}, s.talking())

	// The caller starts talking again, so their words are younger than callerHold.
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "wait one more thing",
	}, s.quiet())

	_, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the caller is talking, so the queued turn waits")
	s.converse.Waiting(s.quiet())
	s.converse.Waiting(s.quiet())
	s.eventually(func() bool { return s.holds() == 1 }, "the hold was never reported")
	s.Equal(1, s.holds(), "a wait spanning several retries is one hold, not one per retry")

	// Once the new words are resolved, the turn is handed back.
	s.ignoreNextSettled()
	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal("and make it eight", action.Candidate.Text)
}

func (s *ConverseSuite) overlaps() []OverlapDecided {
	var kept []OverlapDecided
	for _, event := range s.seen() {
		if typed, ok := event.(OverlapDecided); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func (s *ConverseSuite) decisions() []Decided {
	var kept []Decided
	for _, event := range s.seen() {
		if typed, ok := event.(Decided); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func kinds(actions []Action) []ActionKind {
	found := make([]ActionKind, 0, len(actions))
	for _, action := range actions {
		found = append(found, action.Kind)
	}
	return found
}

func (s *ConverseSuite) TestWhatIsDecidedAboutATurnAndWhoHasTheFloor() {
	// The two halves of a ruling are independent: what the caller said settles whether
	// they get an answer, and whether the agent was talking settles when.
	cases := []struct {
		name        string
		disposition harness.Disposition
		floor       harness.Floor
		speaking    bool
		expected    []ActionKind
		clarify     string
	}{
		{
			name:        "an unfinished thought is left alone",
			disposition: harness.Wait,
			floor:       harness.Continue,
			expected:    []ActionKind{ActWait},
		},
		{
			name:        "speech meant for somebody else is dropped",
			disposition: harness.Ignore,
			floor:       harness.Continue,
			expected:    []ActionKind{ActIgnore},
		},
		{
			name:        "a complete thought is answered",
			disposition: harness.Respond,
			floor:       harness.Continue,
			expected:    []ActionKind{ActAnswer},
		},
		{
			name:        "an ambiguous request is answered with a question",
			disposition: harness.Clarify,
			floor:       harness.Continue,
			expected:    []ActionKind{ActAnswer},
			clarify:     ambiguousNote,
		},
		{
			name:        "a correction takes the floor from the agent",
			disposition: harness.Respond,
			floor:       harness.Stop,
			speaking:    true,
			expected:    []ActionKind{ActInterrupt, ActAnswer},
		},
		{
			name:        "an addition cuts the reply short and waits its turn",
			disposition: harness.Respond,
			floor:       harness.Shorten,
			speaking:    true,
			expected:    []ActionKind{ActQueue, ActShorten},
		},
		{
			name:        "an acknowledgement lets the agent finish first",
			disposition: harness.Respond,
			floor:       harness.Continue,
			speaking:    true,
			expected:    []ActionKind{ActQueue},
		},
		{
			name:        "a turn ignored over the agent's speech leaves the reply alone",
			disposition: harness.Ignore,
			floor:       harness.Stop,
			speaking:    true,
			expected:    []ActionKind{ActIgnore},
		},
	}

	for _, test := range cases {
		s.Run(test.name, func() {
			s.build(DuplexOptions{})

			state := s.quiet()
			if test.speaking {
				state = s.talking()
			}
			ready := s.settle("book a table for four", state)

			actions := s.converse.Ruled(harness.Decided{
				CandidateID: ready.ID,
				Disposition: test.disposition,
				Floor:       test.floor,
				TookMs:      12,
			}, state)

			s.Equal(test.expected, kinds(actions))
			for _, action := range actions {
				if action.Kind == ActAnswer || action.Kind == ActQueue {
					s.Equal(test.clarify, action.Clarify)
					s.Equal(ready.ID, action.Candidate.ID)
				}
				if action.Kind == ActInterrupt || action.Kind == ActShorten {
					s.Equal("turn-1", action.TurnID, "the reply in flight is what is cut")
				}
			}
		})
	}
}

func (s *ConverseSuite) TestACoughOverlappingAReplyIsIgnoredEvenIfTheControllerWouldAnswer() {
	s.build(DuplexOptions{})
	state := s.talking()
	ready := s.settle("cough", state)

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Stop,
	}, state)

	s.Equal([]ActionKind{ActIgnore}, kinds(actions))
}

func (s *ConverseSuite) TestOnlyAnAcceptedTurnCountsAsSomethingTheCallerSaid() {
	// A turn the agent decided not to answer was still heard by the transcriber, and
	// reporting it as heard would put words in the conversation nobody acted on.
	ready := s.settle("mumbling in the background", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Ignore,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Empty(s.heard())

	answered := s.settle("book a table for four", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: answered.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())

	s.eventually(func() bool { return len(s.heard()) == 1 }, "the answered turn was never reported")
	s.Equal("book a table for four", s.heard()[0].Text)
}

func (s *ConverseSuite) TestAnUnfinishedThoughtIsPutAgainOnceTheCallerStops() {
	// Waiting must not lose the turn: the same words come back after a longer pause.
	ready := s.settle("book a table", s.quiet())

	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	select {
	case again := <-s.settling.Ready():
		s.Equal("book a table", again.Text)
		s.NotEqual(ready.ID, again.ID)
	case <-time.After(testWithin):
		s.Fail("the caller lost their turn to a controller that wanted to wait")
	}
}

func (s *ConverseSuite) TestACallerWhoGoesQuietOnAnUnfinishedThoughtIsAskedWhatTheyMeant() {
	// Waiting is a loop only the caller can end, so a thought that never arrives leaves the
	// agent listening to silence for the rest of the call. It is also as likely to be
	// something the transcriber mangled as something the caller gave up on.
	ready := s.settle("book a", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	again := s.held()
	s.converse.Settled(again, s.quiet())
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: again.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Require().Equal([]ActionKind{ActAnswer}, kinds(actions))
	s.Equal(unfinishedNote, actions[0].Clarify)
	s.Equal("book a", actions[0].Candidate.Text)
}

func (s *ConverseSuite) TestWordsThatKeepChangingAreStillWaitedOn() {
	// Only silence runs the patience out. Somebody who is still talking is finishing their
	// thought, and asking them what they meant is interrupting them to do it.
	first := s.settle("could you", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: first.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	time.Sleep(2 * testPatience)
	second := s.settle("could you book", s.quiet())
	actions := s.converse.Ruled(harness.Decided{
		CandidateID: second.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Equal([]ActionKind{ActWait}, kinds(actions))
}

func (s *ConverseSuite) TestWordsThatHaveNotChangedAreOnlyWrittenDownOnce() {
	// The retry puts the same words to the controller for as long as the caller says
	// nothing more, and each lap decides exactly what the last one did. A trail with all of
	// them in it is one nobody can read.
	s.start(DuplexOptions{}, time.Hour)

	ready := s.settle("book a", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())
	s.eventually(func() bool { return len(s.decisions()) == 2 },
		"the first ask and wait were never reported")

	again := s.held()
	s.converse.Settled(again, s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: again.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	}, s.quiet())

	s.Len(s.decisions(), 2, "the same judgement about the same words was written down twice")

	added := s.settle("book a table", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: added.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet())

	s.eventually(func() bool { return len(s.decisions()) > 2 },
		"the caller said something new and nobody wrote it down")
}

func (s *ConverseSuite) TestTheTurnAfterAnOverlapIsGivenLongerToSettle() {
	// Two people talking at once is as often a line running late as a change of mind, so
	// the next thing said is given longer to arrive in full before it is answered.
	ready := s.settle("actually", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Stop,
	}, s.talking())

	started := time.Now()
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "make it nine",
	}, s.quiet())
	s.held()

	s.GreaterOrEqual(time.Since(started), interruptGrace,
		"the turn after an overlap was settled at the usual pace")
}

func (s *ConverseSuite) TestARulingAboutWordsThatHaveChangedIsNotActedOn() {
	ready := s.settle("book a table", s.quiet())

	actions := s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeReplacement,
		Text:        "book a table for four",
	}, s.quiet())

	s.Equal([]ActionKind{ActSupersede}, kinds(actions))
	s.Equal(ready.ID, actions[0].TurnID)

	s.Empty(s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet()), "answering the words as they were would answer half a sentence")
}

func (s *ConverseSuite) TestARulingForATurnThatHasMovedOnIsIgnored() {
	s.Empty(s.converse.Ruled(harness.Decided{
		CandidateID: "turn-gone",
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.quiet()))
}

func (s *ConverseSuite) TestAControllerThatDidNotAnswerDoesNotCostTheCallerTheirTurn() {
	ready := s.settle("book a table", s.quiet())

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: "nonsense",
		Floor:       harness.Continue,
	}, s.quiet())
	s.Equal([]ActionKind{ActAnswer}, kinds(actions))
	s.Equal("book a table", actions[0].Text)
	s.eventually(func() bool { return len(s.heard()) == 1 }, "the caller's words were never recorded")
	s.Equal("book a table", s.heard()[0].Text)
}

func (s *ConverseSuite) TestAnUnfinishedTurnWhoseControllerFailedIsPutAgain() {
	ready := candidate{
		ID:          "turn-unfinished",
		Participant: caller,
		Text:        "book a",
		Unfinished:  true,
	}
	s.Equal(ActAsk, s.converse.Settled(ready, s.quiet()).Kind)

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Err:         errors.New("deadline"),
	}, s.quiet())
	s.Equal([]ActionKind{ActFail}, kinds(actions))
	s.Equal("book a", actions[0].Text)
	s.Error(actions[0].Err)
}

func (s *ConverseSuite) TestAFinishedTurnOverTheAgentWhoseControllerFailedTakesTheFloor() {
	ready := s.settle("make it six people, and gluten instead", s.talking())

	actions := s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Err:         errors.New("402 Payment Required"),
	}, s.talking())
	s.Equal([]ActionKind{ActInterrupt, ActAnswer}, kinds(actions),
		"queueing it would leave the caller waiting out the whole reply")
	s.Equal("make it six people, and gluten instead", actions[1].Text)
}

func (s *ConverseSuite) TestATurnHeldOverTheAgentIsAnsweredOnceItStopsTalking() {
	ready := s.settle("and make it eight o'clock", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
	}, s.talking())

	_, waiting := s.converse.Waiting(s.talking())
	s.False(waiting, "answering while still speaking would have the agent talk over itself")

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal(ActAnswer, action.Kind)
	s.Equal("and make it eight o'clock", action.Candidate.Text)

	_, waiting = s.converse.Waiting(s.quiet())
	s.False(waiting, "a turn is answered once")
}

func (s *ConverseSuite) TestOnlyTheLastThingSaidOverTheAgentIsAnswered() {
	// A caller who says three things while being talked over is owed an answer to the
	// last of them, not a reply to each in turn once the agent finally stops.
	for _, said := range []string{"actually", "make it eight", "make it nine"} {
		ready := s.settle(said, s.talking())
		s.converse.Ruled(harness.Decided{
			CandidateID: ready.ID,
			Disposition: harness.Respond,
			Floor:       harness.Continue,
		}, s.talking())
	}

	action, waiting := s.converse.Waiting(s.quiet())
	s.Require().True(waiting)
	s.Equal("make it nine", action.Candidate.Text)
}

func (s *ConverseSuite) TestWhoKeepsTheFloorIsReported() {
	ready := s.settle("actually", s.talking())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Shorten,
	}, s.talking())

	s.eventually(func() bool { return len(s.overlaps()) == 1 }, "the overlap was never reported")
	s.Equal("shorten", s.overlaps()[0].Action)
	s.Equal("turn-1", s.overlaps()[0].TurnID)
}

func (s *ConverseSuite) TestAMurmurNeedsSomethingWorthAcknowledgingAndAQuietAgent() {
	s.build(DuplexOptions{Backchannel: true, BackchannelWords: 4})

	s.Empty(kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so",
	}, s.quiet())), "acknowledging one word is interrupting")

	s.Equal([]ActionKind{ActBackchannel}, kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so I was wondering whether",
	}, s.quiet())))
}

func (s *ConverseSuite) TestTheAgentDoesNotMurmurOverItself() {
	s.build(DuplexOptions{Backchannel: true, BackchannelWords: 2})

	s.NotContains(kinds(s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeReplacement, Text: "so I was wondering whether",
	}, s.talking())), ActBackchannel, "a murmur over the agent's own reply is not made")
}

func (s *ConverseSuite) TestACallerWhoDoesNotAnswerAQuestionIsCheckedOnSooner() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Asked = true
	state.LastSpokeAt = time.Now().Add(-answerIdleGap - time.Second)

	actions := s.converse.Tick(state)
	s.Require().Equal([]ActionKind{ActCheckIn}, kinds(actions),
		"a caller who owes an answer and says nothing may have stepped away or lost the line")
	s.Equal(answerPurpose, actions[0].Compose)

	state.Asked = false
	s.build(DuplexOptions{})
	s.Empty(s.converse.Tick(state), "after a statement the same silence is somebody thinking")
}

func (s *ConverseSuite) TestALongSilenceWhileWorkRunsSaysWhatIsStillGoing() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Working = "the render"
	state.LastSpokeAt = time.Now().Add(-11 * time.Second)

	actions := s.converse.Tick(state)
	s.Require().Equal([]ActionKind{ActCheckIn}, kinds(actions))
	s.Equal(updatePurpose, actions[0].Compose, "the model writes what is said, not a stock line")
	s.Empty(actions[0].Text)
}

func (s *ConverseSuite) TestAShortSilenceWhileWorkRunsIsLeftAlone() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Working = "the render"
	state.LastSpokeAt = time.Now().Add(-2 * time.Second)

	s.Empty(s.converse.Tick(state))
}

func (s *ConverseSuite) TestACallerWaitingOnWorkIsNeverAskedWhetherThereIsAnythingElse() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Working = "the render"
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	var purposes []string
	for range 5 {
		for _, action := range s.converse.Tick(state) {
			purposes = append(purposes, action.Compose)
		}
	}
	s.Equal([]string{updatePurpose, updatePurpose}, purposes,
		"two updates, then the answer is what they are waiting for")
}

func (s *ConverseSuite) TestNothingMoreIsDecidedWhileALineIsBeingWritten() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Composing = true
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	s.Empty(s.converse.Tick(state), "a second line would only say the first one again")
	state.Working = "the render"
	s.Empty(s.converse.Tick(state))
}

func (s *ConverseSuite) TestAnAnswerOwedToTheCallerIsNotTalkedOverWithACheckIn() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Owed = true
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	s.Empty(s.converse.Tick(state))
}

func (s *ConverseSuite) TestNoUpdateIsSaidOverTheAgentsOwnReply() {
	s.build(DuplexOptions{})

	state := s.quiet()
	state.Working = "the render"
	state.Talking = true
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	s.Empty(s.converse.Tick(state))
}

func (s *ConverseSuite) TestACallNobodyHasSpokenOnIsAskedWhetherAnythingElseIsNeeded() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Hour)

	actions := s.converse.Tick(state)
	s.Require().Equal([]ActionKind{ActCheckIn}, kinds(actions))
	s.Equal(idlePurpose, actions[0].Compose)
}

func (s *ConverseSuite) TestACallThatHasJustGoneQuietIsLeftAlone() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Second)

	s.Empty(s.converse.Tick(state), "a pause is somebody thinking, not a call that has died")
}

func (s *ConverseSuite) TestSomebodyStillTalkingIsNotASilentCall() {
	s.build(DuplexOptions{Backchannel: true, BackchannelGap: time.Millisecond})

	state := s.quiet()
	state.LastSpokeAt = time.Now().Add(-time.Hour)
	state.LastHeardAt = time.Now()

	s.Empty(s.converse.Tick(state), "the caller spoke a moment ago, so the agent owes them an answer")
}

func (s *ConverseSuite) TestTheWordsAreReportedAsTheyArrive() {
	// A watcher sees the caller's sentence build up, which is what makes a live call
	// worth watching rather than waiting out.
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeDelta, Text: "book a ",
	}, s.quiet())
	s.converse.Observe(stt.Transcript{
		Participant: caller, Mode: stt.ModeDelta, Text: "table",
	}, s.quiet())

	s.eventually(func() bool { return len(s.hearing()) == 2 }, "the words were never reported")
	s.Equal("book a", s.hearing()[0].Text)
	s.Equal("book a table", s.hearing()[1].Text,
		"a delta is a piece of a sentence, and the sentence is what somebody reads")
}

func (s *ConverseSuite) hearing() []Hearing {
	var kept []Hearing
	for _, event := range s.seen() {
		if typed, ok := event.(Hearing); ok {
			kept = append(kept, typed)
		}
	}
	return kept
}

func (s *ConverseSuite) TestEveryJudgementIsReportedWithItsReason() {
	// The point of the trail is that a call can be read back. A decision with no reason
	// on it is a line in a log that explains nothing.
	ready := s.settle("book a table for four", s.quiet())
	s.converse.Ruled(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Respond,
		Floor:       harness.Continue,
		TookMs:      42,
	}, s.quiet())

	s.eventually(func() bool { return len(s.decisions()) == 2 }, "the decisions were never reported")

	asked := s.decisions()[0]
	s.Equal(string(ActAsk), asked.Kind)
	s.NotEmpty(asked.Reason)
	s.Equal(ready.ID, asked.TurnID)

	answered := s.decisions()[1]
	s.Equal(string(ActAnswer), answered.Kind)
	s.NotEmpty(answered.Reason)
	s.Equal("book a table for four", answered.Text)
	s.Equal(caller, answered.Participant)
	s.InDelta(42, answered.LatencyMs, 0.001)
	s.False(answered.At.IsZero())
}

func (s *ConverseSuite) TestDelegatedWorkIsRecordedComingBackAsWellAsGoingOut() {
	// A trail that says work went out and never what became of it reads the same whether
	// the subagent answered or ran out of time, which is the one thing worth knowing about
	// a call where the caller never got their answer.
	s.converse.Delegating("task-1", "think", "traffic on I-70", "turn-7")
	s.converse.Delegated(harness.Result{
		TaskID:    "task-1",
		Skill:     "think",
		State:     harness.Done,
		Text:      "The tunnel is clear.",
		ElapsedMs: 4200,
	})

	s.eventually(func() bool { return len(s.decisions()) == 2 }, "the delegation was never reported")

	s.Equal(string(ActDelegate), s.decisions()[0].Kind)

	back := s.decisions()[1]
	s.Equal(string(ActSettle), back.Kind)
	s.Equal("turn-7", back.TurnID, "what came back belongs to the exchange that asked for it")
	s.Equal("The tunnel is clear.", back.Text)
	s.InDelta(4200, back.LatencyMs, 0.001)
	s.Contains(back.Reason, "answered")
}

func (s *ConverseSuite) TestWorkThatWasAbandonedSaysWhyRatherThanGoingQuiet() {
	s.converse.Delegating("task-1", "think", "traffic on I-70", "turn-7")
	s.converse.Delegated(harness.Result{
		TaskID: "task-1",
		Skill:  "think",
		State:  harness.Cancelled,
		Reason: harness.ReasonDeadline,
	})

	s.eventually(func() bool { return len(s.decisions()) == 2 }, "the delegation was never reported")

	back := s.decisions()[1]
	s.Equal(string(ActSettle), back.Kind)
	s.Contains(back.Reason, harness.ReasonDeadline,
		"a caller left without an answer has to be able to read why")
}

func (s *ConverseSuite) TestWorkThatNeedsTheCallerAskedRecordsTheQuestion() {
	s.converse.Delegating("task-1", "think", "best route", "turn-7")
	s.converse.Delegated(harness.Result{
		TaskID:   "task-1",
		Skill:    "think",
		State:    harness.Done,
		Question: "Where are you starting from?",
	})

	s.eventually(func() bool { return len(s.decisions()) == 2 }, "the delegation was never reported")

	back := s.decisions()[1]
	s.Equal("Where are you starting from?", back.Text,
		"the question is the useful half of what came back")
}

func (s *ConverseSuite) TestEmptyNumberedFinalSupersedesItsProvisionalOverlap() {
	const utterance = 91
	var provisional candidate
	for _, action := range s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeDelta,
		Utterance:   utterance,
		Text:        "wait, please",
	}, s.talking()) {
		if action.Kind == ActAsk {
			provisional = action.Candidate
		}
	}
	s.Require().NotEmpty(provisional.ID)

	actions := s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   utterance,
		Text:        "  ",
	}, s.talking())
	var canceled bool
	for _, action := range actions {
		if action.Kind == ActSupersede && action.TurnID == provisional.ID {
			canceled = true
		}
	}
	s.True(canceled, "the provisional ruling must be canceled when its utterance finalizes empty")

	late := s.converse.Ruled(harness.Decided{
		CandidateID: provisional.ID,
		Disposition: harness.Respond,
		Floor:       harness.Stop,
	}, s.talking())
	s.Empty(late, "a stale ruling must not interrupt after its transcript was withdrawn")
}

func (s *ConverseSuite) TestStaleEmptyFinalDoesNotCancelANewerSameTextOverlap() {
	const olderUtterance = 92
	const newerUtterance = 93
	observe := func(utterance int64, text string) (candidate, []Action) {
		var provisional candidate
		actions := s.converse.Observe(stt.Transcript{
			Participant: caller,
			Mode:        stt.ModeDelta,
			Utterance:   utterance,
			Text:        text,
		}, s.talking())
		for _, action := range actions {
			if action.Kind == ActAsk {
				provisional = action.Candidate
			}
		}
		return provisional, actions
	}
	older, _ := observe(olderUtterance, "please stop")
	s.Require().NotEmpty(older.ID)
	newer, actions := observe(newerUtterance, "please stop")
	s.Require().NotEmpty(newer.ID, "a new numbered utterance is distinct even with the same text")
	var canceledOlder bool
	for _, action := range actions {
		if action.Kind == ActSupersede && action.TurnID == older.ID {
			canceledOlder = true
		}
	}
	s.True(canceledOlder)

	stale := s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   olderUtterance,
		Text:        "",
	}, s.talking())
	s.Empty(stale, "an empty final for an older turn leaves the newer overlap in flight")

	late := s.converse.Ruled(harness.Decided{
		CandidateID: newer.ID,
		Disposition: harness.Respond,
		Floor:       harness.Stop,
	}, s.talking())
	s.Equal([]ActionKind{ActInterrupt}, kinds(late))
}

func (s *ConverseSuite) TestNumberedPartialKeepsAnOlderQueuedReplyUntilFinal() {
	const previousUtterance = 95
	const activeUtterance = 96
	queued := s.queueSettledTurn("the earlier request", previousUtterance, harness.Continue)
	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeDelta,
		Utterance:   activeUtterance,
		Text:        "one more thing",
	}, s.quiet())

	// The provider can take longer than callerHold to send its adaptive final preview.
	time.Sleep(callerHold + 20*time.Millisecond)
	got, waiting := s.converse.Waiting(s.quiet())
	s.False(waiting, "the previous answer remains queued while a numbered turn is unfinished")
	s.Empty(got.Kind)
	s.True(s.converse.Listening())

	s.converse.Observe(stt.Transcript{
		Participant: caller,
		Mode:        stt.ModeFinal,
		Utterance:   activeUtterance,
		Text:        "one more thing",
	}, s.quiet())

	got, waiting = s.converse.Waiting(s.quiet())
	s.True(waiting, "the previous answer can drain once the numbered turn has finalized")
	s.Equal(ActAnswer, got.Kind)
	s.Equal(queued.ID, got.Candidate.ID)
}
