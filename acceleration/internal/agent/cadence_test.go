package agent

import (
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type CadenceSuite struct {
	suite.Suite
	cadence *cadence
}

func TestCadenceSuite(t *testing.T) {
	suite.Run(t, new(CadenceSuite))
}

func (s *CadenceSuite) SetupTest() {
	s.cadence = newCadence(5*time.Millisecond, 10*time.Millisecond, 100*time.Millisecond,
		slog.New(slog.DiscardHandler))
	s.T().Cleanup(s.cadence.Close)
}

func (s *CadenceSuite) ready() candidate {
	select {
	case ready := <-s.cadence.Ready():
		return ready
	case <-time.After(time.Second):
		s.FailNow("the transcript never became ready")
		return candidate{}
	}
}

func (s *CadenceSuite) TestAStableRevisionBecomesReadyWithoutAFinalEvent() {
	alice := stt.Participant{ID: "alice"}

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "book a table",
	})
	s.Empty(superseded)
	s.Equal("book a table", saying, "the words so far are what a watcher sees as they arrive")

	ready := s.ready()
	s.Equal(alice, ready.Participant)
	s.Equal("book a table", ready.Text)
	s.NotEmpty(ready.ID)
}

// quiet asserts nothing becomes ready, which is how "the agent does not answer that"
// looks from here.
func (s *CadenceSuite) quiet() {
	select {
	case ready := <-s.cadence.Ready():
		s.Failf("nothing should have been ready", "got %q", ready.Text)
	case <-time.After(50 * time.Millisecond):
	}
}

func (s *CadenceSuite) TestTheFinalCopyOfAnAnsweredUtteranceIsNotAnsweredAgain() {
	// The transcriber settles on words the agent has already started answering, which is
	// not the caller saying them a second time.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "how is your day going",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeFinal,
		Text:        "How is your day going?",
	})
	s.Empty(superseded)
	s.Empty(saying, "the transcriber going over itself is not the caller saying anything")

	s.quiet()
}

func (s *CadenceSuite) TestAnAnsweredUtteranceRespelledOnTheWayOutIsNotAnsweredAgain() {
	// Gemini streams an order number as digits and commits it as words. Both are the same
	// speech, so answering the second one asks the caller for the number they just gave.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "1 2 3",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeFinal, Utterance: 2, Text: "one two three",
	})

	s.Empty(superseded)
	s.Empty(saying, "the transcriber respelling itself is not the caller saying anything")
	s.quiet()
}

func (s *CadenceSuite) TestWordsAddedToAnAnsweredUtteranceAreStillHeard() {
	// Answering part of what somebody is saying must not cost them the rest of it, so a
	// run of speech that grows after an answer is new words rather than a respelling.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "order 1 2",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "order 1 2 3 4",
	})

	s.Equal("order 1 2 3 4", s.ready().Text)
}

func (s *CadenceSuite) TestSayingTheSameThingAgainLaterIsHeardAgain() {
	// Repeating yourself is a normal thing to do in a conversation, so only the
	// transcriber's immediate restatement is discarded.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "yes"})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))
	time.Sleep(150 * time.Millisecond)

	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "yes"})

	repeated := s.ready()
	s.Equal("yes", repeated.Text)
	s.NotEqual(answered.ID, repeated.ID)
}

func (s *CadenceSuite) TestOneUtteranceIsAnsweredOnceHoweverLongTheTranscriberGoesOverIt() {
	// Deepgram Flux restates a word it has settled on for as long as the track is open,
	// which outlasts any wall clock and had the agent answering a single hello six times.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "hey",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	for range 3 {
		time.Sleep(80 * time.Millisecond)
		superseded, _ := s.cadence.Observe(stt.Transcript{
			Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "hey",
		})
		s.Empty(superseded)
	}

	s.quiet()
}

func (s *CadenceSuite) TestTheSameWordSaidAgainInANewUtteranceIsAnswered() {
	// Somebody saying "hey" a second time is owed a second answer, even straight away,
	// which is what the transcriber's own count settles and no amount of waiting can.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "hey",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "hey",
	})

	repeated := s.ready()
	s.Equal("hey", repeated.Text)
	s.NotEqual(answered.ID, repeated.ID)
}

func (s *CadenceSuite) TestNewWordsAfterAnAnswerAreHeard() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "book a table",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "for four people",
	})

	s.Equal("for four people", s.ready().Text)
}

func (s *CadenceSuite) TestNewWordsSupersedeAControllerDecision() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "book a table",
	})
	first := s.ready()

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "book a table for four",
	})

	s.Equal(first.ID, superseded)
	s.Equal("book a table for four", saying)
	s.Equal("book a table for four", s.ready().Text)
}

func (s *CadenceSuite) TestAFinalCopyDoesNotDriveOrDelayCadence() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "book a table",
	})
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeFinal,
		Text:        "Book a table.",
	})

	s.Equal("book a table", s.ready().Text)
}

func (s *CadenceSuite) TestWaitingRetriesUnchangedWords() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "could you",
	})
	first := s.ready()

	s.True(s.cadence.Resolve(first.ID, true))

	retried := s.ready()
	s.NotEqual(first.ID, retried.ID)
	s.Equal(first.Text, retried.Text)
}

func (s *CadenceSuite) TestGraceGivesOneTurnLongerToHoldStill() {
	// The turn after an overlap waits longer, because the line is running late and the rest
	// of the sentence is still on its way. The call is not slow for having had one
	// collision in it, so the next turn is settled at the usual pace.
	alice := stt.Participant{ID: "alice"}
	grace := 60 * time.Millisecond
	s.cadence.Grace(grace)

	started := time.Now()
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Text: "book a table",
	})
	first := s.ready()
	s.GreaterOrEqual(time.Since(started), grace)
	s.Require().True(s.cadence.Resolve(first.ID, false))

	started = time.Now()
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Text: "for four people",
	})
	s.ready()

	s.Less(time.Since(started), grace, "the grace was for one turn, not for the rest of the call")
}

func (s *CadenceSuite) TestDeltasAreAccumulated() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeDelta, Text: "hello "})
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeDelta, Text: "there"})

	s.Equal("hello there", s.ready().Text)
}

func (s *CadenceSuite) TestParticipantsKeepIndependentCadences() {
	alice := stt.Participant{ID: "alice"}
	bob := stt.Participant{ID: "bob"}
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "hello"})
	s.cadence.Observe(stt.Transcript{Participant: bob, Mode: stt.ModeReplacement, Text: "background"})

	first := s.ready()
	second := s.ready()
	s.ElementsMatch([]string{"alice", "bob"}, []string{first.Participant.ID, second.Participant.ID})
}
