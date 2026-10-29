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

func (s *CadenceSuite) TestOnlyATranscriberThatMarksEndsHasTurnsThatHaveNotEnded() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "a table for four"})
	s.True(s.ready().Ended, "a transcriber that never marks an end is not waited on for one")
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "a table for four please"})
	s.True(s.ready().Ended)

	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "a table for four please and"})

	s.False(s.ready().Ended, "the words grew after the end was marked")
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

func (s *CadenceSuite) TestWordsAddedToAnAnsweredRunAreSaidWithoutTheAnsweredOnes() {
	// Deepgram Flux resumes a turn it already ended and sends the whole run each time, so
	// passing the run on would hand the model every earlier answer again, more each turn.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1,
		Text: "Sitting for a long time makes it worse.",
	})
	first := s.ready()
	s.Equal("Sitting for a long time makes it worse.", first.Said())
	s.Require().True(s.cadence.Resolve(first.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1,
		Text: "Sitting for a long time makes it worse. Walking helps.",
	})
	second := s.ready()
	s.Equal("Sitting for a long time makes it worse. Walking helps.", second.Text,
		"turn-taking still compares the whole run")
	s.Equal("Walking helps.", second.Said())
	s.Require().True(s.cadence.Resolve(second.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1,
		Text: "Sitting for a long time makes it worse. Walking helps. It wakes me up at night.",
	})
	s.Equal("It wakes me up at night.", s.ready().Said())
}

func (s *CadenceSuite) TestANumberReadOutInPiecesIsSaidWhole() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "order 1 2",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "order 1 2 3 4",
	})

	s.Equal("order 1 2 3 4", s.ready().Said(), "3 4 alone is not the order number the caller read out")
}

func (s *CadenceSuite) TestANumberSaidAsWordsIsSaidWhole() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 3, Text: "order one two",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 3, Text: "order one two three four",
	})

	s.Equal("order one two three four", s.ready().Said())
}

func (s *CadenceSuite) TestLettersSpelledOutAreSaidWhole() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 4, Text: "my member ID is A B C",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 4, Text: "my member ID is A B C 1 2 3",
	})

	s.Equal("my member ID is A B C 1 2 3", s.ready().Said())
}

func (s *CadenceSuite) TestAnIOrAnAIsAWordNotALetterBeingSpelled() {
	// Answered on "It came on gradually I", the run that goes on is new words, not a spelling
	// to keep whole; local run of 2026-10-20 repeated the whole run after the lone "I".
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 5, Text: "It came on gradually I",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 5, Text: "It came on gradually I sit at a desk all day",
	})

	s.Equal("sit at a desk all day", s.ready().Said())
}

func (s *CadenceSuite) TestARunTheTranscriberDoesNotNumberIsSaidWhole() {
	// Without a number the same words could be the caller starting again, so nothing is
	// taken as already answered.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "yes"})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "yes please"})

	s.Equal("yes please", s.ready().Said())
}

func (s *CadenceSuite) TestAFinalThatGrowsAnAnsweredIdentifierIsStillHeard() {
	// A final of the same utterance that adds a digit to an identifier already answered is
	// the identifier growing, not a restatement of what was answered.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1,
		Text: "Ines Moreau member ID KTR58302",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeFinal, Utterance: 1,
		Text: "Ines Moreau member ID KTR583026",
	})

	grown := s.ready()
	s.Equal("Ines Moreau member ID KTR583026", grown.Text)
	s.Equal("Ines Moreau member ID KTR583026", grown.Said(), "a longer last token is not new words")
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

func (s *CadenceSuite) TestAClockTimeSplitAcrossUtterancesIsHeardAsOneTurn() {
	// A transcriber can end an utterance after "6:00." and open the next with "fifteen",
	// splitting one clock time across two utterances. The turn is both of them, not the
	// tail alone.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Hi, could I get a table for three this Friday at 6:00.",
	})
	first := s.ready()

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   2,
		Text:        "fifteen somewhere quiet if you can, it's a birthday dinner",
	})

	s.Equal(first.ID, superseded)
	s.Contains(saying, "table for three")
	s.Contains(saying, "fifteen somewhere quiet")
	s.Equal(saying, s.ready().Text)
}

func (s *CadenceSuite) TestWordsCarriedIntoANewUtteranceSurviveItsRevisions() {
	// Flux finalizes "Last name Quillon" and starts the callback number as a new
	// utterance, then revises that utterance on its own. Keeping the name for only the
	// first revision is how the agent asked for a name it had been given.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeFinal, Utterance: 1, Text: "Last name Quillon, q u i l l o n.",
	})
	s.ready()
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Callback is",
	})

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Callback is five one two",
	})

	s.Equal("Last name Quillon, q u i l l o n. Callback is five one two", saying)
	s.Equal(saying, s.ready().Text)
}

func (s *CadenceSuite) TestTheTailOfACarriedTurnRestatedIsNotAnsweredAgain() {
	// A test call answered "Is this all working Wow. Okay. Okay.", carried across two
	// utterances, then heard the second utterance's "Wow. Okay. Okay." again as Flux went
	// over it, talked over its own reply with it and answered it a second time.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "Is this all working",
	})
	s.ready()
	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Wow. Okay. Okay.",
	})
	s.Require().Equal("Is this all working Wow. Okay. Okay.", saying)
	answered := s.ready()
	s.Equal(int64(2), answered.Utterance, "a candidate says which run of speech it came from")
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Wow. Okay. Okay.",
	})

	s.Empty(superseded)
	s.Empty(saying, "the transcriber going over the words it added is not the caller saying them again")
	s.quiet()
}

func (s *CadenceSuite) TestGoingBackOverTheStartOfAnAnsweredUtteranceIsNotNew() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "wow okay okay",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "Wow. Okay.",
	})

	s.Empty(saying, "a revision that says less than the answered turn adds nothing")
	s.quiet()
}

func (s *CadenceSuite) TestAWordThatOnlyStartsLikeAnAnsweredOneIsHeard() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "is this working",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1, Text: "I",
	})

	s.Equal("I", saying, `"I" does not open "is this working": words are compared whole`)
}

func (s *CadenceSuite) TestMoreSaidAfterAnAnsweredUtteranceSaysWhatItRepeats() {
	// Flux sends a run of speech whole: the question answered, then what the caller
	// said over the answer, as one revision of the same utterance.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 3,
		Text: "What is the weather like in Amsterdam today?",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 3,
		Text: "What is the weather like in Amsterdam today? What about tomorrow?",
	})

	s.Equal("What is the weather like in Amsterdam today? What about tomorrow?", saying,
		"the turn is answered on everything the caller said")
	repeats := s.cadence.Repeating("alice")
	s.Equal("What is the weather like in Amsterdam today?", repeats)
	s.Equal("What about tomorrow?", unrepeated(s.ready().Text, repeats))
}

func (s *CadenceSuite) TestTheSameWordsInANewUtteranceRepeatNothing() {
	// The caller asking again is a line of its own, whole.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 3, Text: "Where is my order?",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 4, Text: "Where is my order? It is late.",
	})

	s.Empty(s.cadence.Repeating("alice"))
}

func TestWhatALineSaysWithoutWhatWasAlreadyHeard(t *testing.T) {
	for _, tc := range []struct{ text, repeats, want string }{
		{"Okay. Okay. Thanks.", "Okay.", "Okay. Thanks."},
		{"What is it like today? And tomorrow?", "what is it like today", "And tomorrow?"},
		{"Member ID KTR583027", "member id ktr58302", "Member ID KTR583027"},
		{"Hello there", "", "Hello there"},
		{"Okay.", "Okay", ""},
	} {
		if got := unrepeated(tc.text, tc.repeats); got != tc.want {
			t.Errorf("unrepeated(%q, %q) = %q, want %q", tc.text, tc.repeats, got, tc.want)
		}
	}
}

func (s *CadenceSuite) TestASameUtteranceCorrectionReplacesRatherThanConcatenates() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Friday at 6:00",
	})
	s.ready()

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Friday at 6:15 by the window",
	})
	s.Equal("Friday at 6:15 by the window", saying)
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

func (s *CadenceSuite) TestAFinalSettlesSoonerThanTheGap() {
	// A transcriber that finalizes has already heard the caller stop, so the words are put
	// at once rather than after the gap a revision still waits.
	patient := newCadence(400*time.Millisecond, 800*time.Millisecond, time.Second, slog.New(slog.DiscardHandler))
	s.T().Cleanup(patient.Close)
	alice := stt.Participant{ID: "alice"}
	patient.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeReplacement, Text: "book a table"})
	patient.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "Book a table."})

	select {
	case ready := <-patient.Ready():
		s.Equal("book a table", ready.Text)
	case <-time.After(250 * time.Millisecond):
		s.Fail("a finalized turn should not wait out the whole gap")
	}
}

func (s *CadenceSuite) TestAFinalEndingOnDigitsStillWaitsForThemToGrow() {
	patient := newCadence(400*time.Millisecond, 800*time.Millisecond, time.Second, slog.New(slog.DiscardHandler))
	s.T().Cleanup(patient.Close)
	alice := stt.Participant{ID: "alice"}
	patient.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "my member id is KTR58302"})

	select {
	case ready := <-patient.Ready():
		s.Failf("an identifier may still be growing", "got %q", ready.Text)
	case <-time.After(250 * time.Millisecond):
	}
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

func (s *CadenceSuite) TestAnIncompleteIdentifierWaitsTheRetryGap() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Text:        "member ID KTR58302",
	})
	select {
	case ready := <-s.cadence.Ready():
		s.FailNowf("an incomplete identifier became ready on the short gap", "got %q", ready.Text)
	case <-time.After(7 * time.Millisecond):
	}
	s.Equal("member ID KTR58302", s.ready().Text)
}

func (s *CadenceSuite) TestIncompleteIdentifiersAreTheOnesThatStillHaveADigitTail() {
	s.True(incompleteIdentifier("member ID KTR58302"))
	s.True(incompleteIdentifier("Friday at 6:15"))
	s.True(incompleteIdentifier("callback 512-555-0178"))
	s.True(incompleteIdentifier("PIN 3825"))
	s.False(incompleteIdentifier("book a table"))
	s.False(incompleteIdentifier("party of 4"))
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
