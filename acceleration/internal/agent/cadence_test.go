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

func (s *CadenceSuite) TestAFinalThatGrowsAnAnsweredIdentifierIsStillHeard() {
	// Healthcare golden lost the last digit of ABC123456 because Gemini's final of the
	// same utterance was dropped as a restatement.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 1,
		Text: "Maya Chen member ID ABC12345",
	})
	answered := s.ready()
	s.Require().True(s.cadence.Resolve(answered.ID, false))

	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeFinal, Utterance: 1,
		Text: "Maya Chen member ID ABC123456",
	})

	s.Equal("Maya Chen member ID ABC123456", s.ready().Text)
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
	// Gemini endpointing invents a period after "7:00" and emits "thirty patio..." as a
	// new utterance. Answering only the tail is how the agent booked a party of one.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Hi, I'd like to book the table for four this Saturday at 7:00.",
	})
	first := s.ready()

	superseded, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   2,
		Text:        "thirty patio if you have it, a high chair, and one of us has a peanut allergy",
	})

	s.Equal(first.ID, superseded)
	s.Contains(saying, "table for four")
	s.Contains(saying, "thirty patio")
	s.Equal(saying, s.ready().Text)
}

func (s *CadenceSuite) TestWordsCarriedIntoANewUtteranceSurviveItsRevisions() {
	// Flux finalizes "Last name Alvarez" and starts the callback number as a new
	// utterance, then revises that utterance on its own. Keeping the name for only the
	// first revision is how the agent asked for a name it had been given.
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeFinal, Utterance: 1, Text: "Last name Alvarez, a l v a r e z.",
	})
	s.ready()
	s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Callback is",
	})

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice, Mode: stt.ModeReplacement, Utterance: 2, Text: "Callback is five one two",
	})

	s.Equal("Last name Alvarez, a l v a r e z. Callback is five one two", saying)
	s.Equal(saying, s.ready().Text)
}

func (s *CadenceSuite) TestASameUtteranceCorrectionReplacesRatherThanConcatenates() {
	alice := stt.Participant{ID: "alice"}
	s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Saturday at 7:00",
	})
	s.ready()

	_, saying := s.cadence.Observe(stt.Transcript{
		Participant: alice,
		Mode:        stt.ModeReplacement,
		Utterance:   1,
		Text:        "Saturday at 7:30 patio",
	})
	s.Equal("Saturday at 7:30 patio", saying)
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
	patient.Observe(stt.Transcript{Participant: alice, Mode: stt.ModeFinal, Text: "my member id is ABC12345"})

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
		Text:        "member ID ABC12345",
	})
	select {
	case ready := <-s.cadence.Ready():
		s.FailNowf("an incomplete identifier became ready on the short gap", "got %q", ready.Text)
	case <-time.After(7 * time.Millisecond):
	}
	s.Equal("member ID ABC12345", s.ready().Text)
}

func (s *CadenceSuite) TestIncompleteIdentifiersAreTheOnesThatStillHaveADigitTail() {
	s.True(incompleteIdentifier("member ID ABC12345"))
	s.True(incompleteIdentifier("Saturday at 7:30"))
	s.True(incompleteIdentifier("callback 512-555-0142"))
	s.True(incompleteIdentifier("PIN 4471"))
	s.False(incompleteIdentifier("book a table"))
	s.False(incompleteIdentifier("party of 4"))
	s.False(incompleteIdentifier("a burger, and"), "words that are still going are not identifiers")
	s.False(incompleteIdentifier("name,"))
}

func (s *CadenceSuite) TestUnfinishedWordsEndOnACommaAConjunctionOrAHesitation() {
	for _, text := range []string{
		"Burger, no bun,", "Name,", "Burger, no bun, ", "你好，", "これ、", "مرحبا،",
		"a burger and", "Or", "I want that but", "it is late because", "I would like, um", "uh", "er",
		"a burger and.", "AND...", "um…",
		`He said "no bun,"`, "she said ‘no bun,’", "«no bun,»", "「你好，」",
	} {
		s.Truef(visiblyUnfinished(text, ""), "%q is a caller part way through", text)
	}
	for _, text := range []string{
		"", "book a table", "Book a table.", "a rock band", "next summer", "the doctor", "my brother",
		"I said uh-huh", "a burger, no bun", "yes, please.", "PIN 4471", "that is all, thanks",
		"I think so.", "I think so", "So", "Is that so?", `He said "no bun"`,
	} {
		s.Falsef(visiblyUnfinished(text, ""), "%q is not still going", text)
	}
}

func (s *CadenceSuite) TestTheWordsAThinkingSpeakerLeavesASentenceOnAreEnglish() {
	for _, language := range []string{"", "en", "EN", "en-US", "en-GB"} {
		s.Truef(visiblyUnfinished("a burger and", language), "%q is English or unsaid", language)
		s.Truef(visiblyUnfinished("I would like, um", language), "%q is English or unsaid", language)
	}
	for _, language := range []string{"de", "es", "fr-CA", "ja", "eng", "end"} {
		s.Falsef(visiblyUnfinished("um", language), "an \"um\" in %q is not a hesitation", language)
		s.Falsef(visiblyUnfinished("or", language), "an \"or\" in %q is not a conjunction", language)
		s.Truef(visiblyUnfinished("pan, queso,", language), "a comma is a comma in %q", language)
		s.Truef(visiblyUnfinished("你好，", language), "a comma is a comma in %q", language)
	}
}

// settleDelay is how long a transcript of the given kind is made to wait on the default
// pacing, read off the timer it schedules rather than waited out.
func (s *CadenceSuite) settleDelay(mode stt.Mode, text string) time.Duration {
	return s.settleDelayIn("", mode, text)
}

// settleDelayIn is settleDelay for a transcript in the given language.
func (s *CadenceSuite) settleDelayIn(language string, mode stt.Mode, text string) time.Duration {
	s.useDefaultCadence()
	timers := s.captureTimers()
	s.cadence.Observe(stt.Transcript{
		Participant: stt.Participant{ID: "caller"}, Mode: mode, Text: text, Language: language,
	})
	s.Require().Len(*timers, 1)
	return (*timers)[0].delay
}

func (s *CadenceSuite) TestAFinalEndingOnACommaWaitsTheRetryGap() {
	// A caller reading out an order stops after every item, and the transcriber finalizes
	// each stop. Answering the first of them is how the agent talks over the rest.
	for _, text := range []string{"Burger, no bun,", "Name,", "你好，"} {
		s.Equalf(defaultCadenceRetry, s.settleDelay(stt.ModeFinal, text), "%q", text)
	}
	s.Equal(defaultCadenceRetry, s.settleDelay(stt.ModeReplacement, "Burger, no bun,"),
		"words that are still going wait the longer gap whether or not they are final")
}

func (s *CadenceSuite) TestAFinalEndingOnAConjunctionOrHesitationWaitsTheRetryGap() {
	for _, text := range []string{"a burger and", "Or", "maybe but", "late because", "I would like, um", "uh.", "Er"} {
		s.Equalf(defaultCadenceRetry, s.settleDelay(stt.ModeFinal, text), "%q", text)
	}
}

func (s *CadenceSuite) TestAFinalEndingOnSoSettlesAtOnce() {
	// "so" closes a sentence as often as it joins one.
	for _, text := range []string{"I think so.", "I think so", "Is that so?", "So"} {
		s.Equalf(cadenceFinalGap, s.settleDelay(stt.ModeFinal, text), "%q", text)
	}
}

func (s *CadenceSuite) TestAFinalEndingOnAnEnglishFillerIsOnlyHeldInEnglish() {
	for _, language := range []string{"", "en", "en-US"} {
		s.Equalf(defaultCadenceRetry, s.settleDelayIn(language, stt.ModeFinal, "um"), "%q", language)
	}
	for _, language := range []string{"de", "pt-BR", "ja"} {
		s.Equalf(cadenceFinalGap, s.settleDelayIn(language, stt.ModeFinal, "um"), "%q", language)
		s.Equalf(cadenceFinalGap, s.settleDelayIn(language, stt.ModeFinal, "a burger or"), "%q", language)
		s.Equalf(defaultCadenceRetry, s.settleDelayIn(language, stt.ModeFinal, "pan, queso,"), "%q", language)
	}
}

func (s *CadenceSuite) TestAFinalEndingOnACommaBeforeAClosingQuoteWaitsTheRetryGap() {
	for _, text := range []string{`He said "no bun,"`, "«sin cebolla,»"} {
		s.Equalf(defaultCadenceRetry, s.settleDelayIn("es", stt.ModeFinal, text), "%q", text)
	}
}

func (s *CadenceSuite) TestAFinalEndingOnAWordThatOnlyContainsOneSettlesAtOnce() {
	// Only whole words count: "band" is not "and" and "summer" is not "um".
	for _, text := range []string{"a rock band", "next summer", "the doctor", "my brother"} {
		s.Equalf(cadenceFinalGap, s.settleDelay(stt.ModeFinal, text), "%q", text)
	}
}

func (s *CadenceSuite) TestAFinalEndingOnAPeriodStillSettlesAtOnce() {
	s.Equal(cadenceFinalGap, s.settleDelay(stt.ModeFinal, "Book a table."))
	s.Equal(defaultCadenceGap, s.settleDelay(stt.ModeReplacement, "Book a table"),
		"a revision that is not final keeps the usual gap")
}

func (s *CadenceSuite) TestTheSameUnfinishedWordsFinalizedAgainDoNotShortenTheWait() {
	s.useDefaultCadence()
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}

	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "Burger, no bun,"})
	s.Require().Len(*timers, 1)
	s.Equal(defaultCadenceRetry, (*timers)[0].delay)

	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "Burger, no bun,"})
	s.Len(*timers, 1, "the final of words that are still going is not a reason to reschedule them")
	s.False((*timers)[0].stopped)

	// The transcriber punctuating on the way out is the same: the words have not changed,
	// and what they now say is that there is more to come, not less.
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "Name"})
	s.Require().Len(*timers, 2)
	s.Equal(defaultCadenceGap, (*timers)[1].delay)
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "Name,"})
	s.Len(*timers, 2)
	s.False((*timers)[1].stopped)
}

func (s *CadenceSuite) TestTheSameFinishedWordsFinalizedAgainAreStillSettledAtOnce() {
	s.useDefaultCadence()
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}

	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "book a table"})
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "Book a table."})

	s.Require().Len(*timers, 2)
	s.True((*timers)[0].stopped)
	s.Equal(cadenceFinalGap, (*timers)[1].delay)
}

func (s *CadenceSuite) TestAnUnfinishedFinalReleasesTheTurnOnceTheRetryGapHasPassed() {
	s.useDefaultCadence()
	timers := s.captureTimers()
	s.cadence.Observe(stt.Transcript{
		Participant: stt.Participant{ID: "caller"}, Mode: stt.ModeFinal, Text: "Burger, no bun,",
	})
	s.Require().Len(*timers, 1)
	s.quiet()

	(*timers)[0].fire()
	s.Equal("Burger, no bun,", s.ready().Text)
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

type capturedCadenceTimer struct {
	delay    time.Duration
	callback func()
	stopped  bool
}

func (timer *capturedCadenceTimer) Stop() bool {
	wasActive := !timer.stopped
	timer.stopped = true
	return wasActive
}

func (timer *capturedCadenceTimer) fire() { timer.callback() }

func (s *CadenceSuite) captureTimers() *[]*capturedCadenceTimer {
	var timers []*capturedCadenceTimer
	s.cadence.after = func(delay time.Duration, callback func()) cadenceTimer {
		timer := &capturedCadenceTimer{delay: delay, callback: callback}
		timers = append(timers, timer)
		return timer
	}
	return &timers
}

func (s *CadenceSuite) useDefaultCadence() {
	s.cadence.Close()
	s.cadence = newCadence(defaultCadenceGap, defaultCadenceRetry, defaultCadenceSettle,
		slog.New(slog.DiscardHandler))
	s.T().Cleanup(s.cadence.Close)
}

func (s *CadenceSuite) TestOrdinaryCandidateRetainsConfiguredInitialGap() {
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "hello"})
	s.Require().Len(*timers, 1)
	s.Equal(5*time.Millisecond, (*timers)[0].delay,
		"the final-only fast path is opt-in at the agent callsite, not a global cadence change")
}

func (s *CadenceSuite) TestPrimaryFinalExpeditesOnceButWaitRetryKeepsItsFullGap() {
	s.useDefaultCadence()
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	interim := stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "book a table"}
	s.cadence.Observe(interim)
	s.Require().Len(*timers, 1)
	s.Equal(defaultCadenceGap, (*timers)[0].delay)

	final := interim
	final.Mode = stt.ModeFinal
	s.True(s.cadence.ExpediteFinal(final), "a settled primary candidate should not wait for the initial gap")
	ready := s.ready()
	s.Equal(interim.Text, ready.Text)
	(*timers)[0].fire()
	s.quiet()

	s.Require().True(s.cadence.Resolve(ready.ID, true))
	s.Require().Len(*timers, 2)
	s.Equal(defaultCadenceRetry, (*timers)[1].delay)
	s.False(s.cadence.ExpediteFinal(final), "duplicate final events must not bypass the retry gap")
	(*timers)[1].fire()
	s.Equal(interim.Text, s.ready().Text)
}

func (s *CadenceSuite) TestAFinalThatIsStillGoingIsNotExpedited() {
	// A comma or a joining word says the caller has more to read out, so the primary
	// scorer is not asked early and the retry wait stands.
	caller := stt.Participant{ID: "caller"}
	for _, text := range []string{"Burger, no bun,", "a burger and", "I would like, um"} {
		s.useDefaultCadence()
		timers := s.captureTimers()
		final := stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: text}
		s.cadence.Observe(final)
		s.Require().Len(*timers, 1, "%q", text)
		s.Equal(defaultCadenceRetry, (*timers)[0].delay, "%q", text)

		s.False(s.cadence.ExpediteFinal(final), "%q is not done", text)
		s.False((*timers)[0].stopped, "the wait for %q stays", text)
		s.quiet()

		(*timers)[0].fire()
		s.Equal(text, s.ready().Text)
	}
}

func (s *CadenceSuite) TestAFinalThatAddsACommaToSettledWordsIsNotExpedited() {
	// The cadence holds the words as first heard, and the final may be the first to say
	// there is more to come.
	s.useDefaultCadence()
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "Name"})
	final := stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "Name,"}
	s.cadence.Observe(final)

	s.False(s.cadence.ExpediteFinal(final))
	s.Require().Len(*timers, 1)
	s.False((*timers)[0].stopped)
	s.quiet()
}

func (s *CadenceSuite) TestAFinalEndingOnAPeriodIsStillExpedited() {
	s.useDefaultCadence()
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "book a table"})
	final := stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "Book a table."}
	s.cadence.Observe(final)

	s.True(s.cadence.ExpediteFinal(final))
	s.Equal("book a table", s.ready().Text)
	for _, timer := range *timers {
		timer.fire()
	}
	s.quiet()
}

func (s *CadenceSuite) TestChangedFinalCanExpediteAndEscapedTimerCannotReleaseIt() {
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "book a table"})
	oldTimer := (*timers)[0]

	final := stt.Transcript{Participant: caller, Mode: stt.ModeFinal, Text: "book a table for four"}
	s.cadence.Observe(final)
	newTimer := (*timers)[1]
	s.True(s.cadence.ExpediteFinal(final))
	first := s.ready()
	oldTimer.fire()
	newTimer.fire()
	s.quiet()
	s.Equal(final.Text, first.Text)
}

func (s *CadenceSuite) TestEscapedTimerAfterForgetCannotReleaseRejoinedParticipant() {
	timers := s.captureTimers()
	caller := stt.Participant{ID: "caller"}
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "old words"})
	oldTimer := (*timers)[0]
	s.cadence.Forget(caller)
	s.cadence.Observe(stt.Transcript{Participant: caller, Mode: stt.ModeReplacement, Text: "new words"})
	newTimer := (*timers)[1]

	oldTimer.fire()
	s.quiet()
	newTimer.fire()
	s.Equal("new words", s.ready().Text)
}
