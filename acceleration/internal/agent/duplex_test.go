package agent

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type DuplexSuite struct {
	suite.Suite
}

func TestDuplexSuite(t *testing.T) {
	suite.Run(t, new(DuplexSuite))
}

func (s *DuplexSuite) TestPunctuationAndCaseDoNotChangeWhatWasSaid() {
	// A provisional transcript and the settled one differ in punctuation far more often
	// than in words, and the reply to both is the same reply.
	s.True(sameWords("book a table for four", "Book a table for four."))
	s.True(sameWords("its  ready", "It's ready!"))
	s.False(sameWords("book a table for four", "book a table for five"))
	s.False(sameWords("book a table", "book a table tonight"))
}

func (s *DuplexSuite) TestAMurmurNeedsSomethingWorthAcknowledging() {
	// Acknowledging three words is interrupting, not listening.
	listener := newDuplex(DuplexOptions{Backchannel: true, BackchannelWords: 5})
	alice := stt.Participant{ID: "alice"}

	s.Empty(listener.Heard(alice, "so I was", true))
	s.NotEmpty(listener.Heard(alice, "so I was wondering whether", true))
}

func (s *DuplexSuite) TestTheAgentDoesNotMurmurOverItself() {
	listener := newDuplex(DuplexOptions{Backchannel: true, BackchannelWords: 3})
	alice := stt.Participant{ID: "alice"}

	s.Empty(listener.Heard(alice, "so I was wondering whether", false),
		"talking over someone to tell them you are listening is not listening")
}

func (s *DuplexSuite) TestMurmursAreSpacedOut() {
	// A listener who says "mhm" every other second is heckling.
	listener := newDuplex(DuplexOptions{Backchannel: true, BackchannelWords: 3, BackchannelGap: time.Hour})
	alice := stt.Participant{ID: "alice"}

	s.NotEmpty(listener.Heard(alice, "so I was wondering whether", true))
	s.Empty(listener.Heard(alice, "so I was wondering whether you could", true))
}

func (s *DuplexSuite) TestMurmursDoNotRepeatThemselves() {
	listener := newDuplex(DuplexOptions{Backchannel: true, BackchannelWords: 3, BackchannelGap: time.Nanosecond})
	alice := stt.Participant{ID: "alice"}

	first := listener.Heard(alice, "one two three", true)
	second := listener.Heard(alice, "one two three four", true)

	s.NotEqual(first, second, "saying the same noise twice running sounds like a machine")
}

func (s *DuplexSuite) TestWithoutBackchannelsNothingIsMurmured() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}

	s.Empty(listener.Heard(alice, "so I was wondering whether you could help me with this", true))
}

func (s *DuplexSuite) TestTheLatestRevisionIsKeptEvenWithoutBackchannels() {
	// A provisional end of turn carries no words, so the last revision is what a reply
	// guessed at it would be answering.
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}

	listener.Heard(alice, "book a table", true)

	s.Equal("book a table", listener.Interim(alice))
}

func (s *DuplexSuite) TestAGuessIsMadeOnAProvisionalEndOfTurn() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}

	turnID, abandoned, ok := listener.Eager(alice, "book a table")

	s.True(ok)
	s.True(strings.HasPrefix(turnID, speculationPrefix))
	s.Empty(abandoned, "there was nothing in flight to replace")
}

func (s *DuplexSuite) TestWithoutSpeculationNoGuessIsMade() {
	listener := newDuplex(DuplexOptions{Backchannel: true})
	alice := stt.Participant{ID: "alice"}

	_, _, ok := listener.Eager(alice, "book a table")

	s.False(ok)
}

func (s *DuplexSuite) TestTheSameWordsAreNotGuessedAtTwice() {
	// Guessing twice at one sentence would be paying for the same reply twice.
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	listener.Eager(alice, "book a table")

	_, _, ok := listener.Eager(alice, "Book a table.")

	s.False(ok)
}

func (s *DuplexSuite) TestASecondGuessReplacesTheFirst() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	first, _, _ := listener.Eager(alice, "book a table")

	_, abandoned, ok := listener.Eager(alice, "book a table for four")

	s.True(ok)
	s.Equal(first, abandoned, "the words it answered are not what they said")
}

func (s *DuplexSuite) TestATurnThatSettlesOnTheGuessedWordsPromotesIt() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	guess, _, _ := listener.Eager(alice, "book a table for four")

	promoted, abandoned := listener.Settled(alice, "Book a table for four.")

	s.Equal(guess, promoted)
	s.Empty(abandoned)
}

func (s *DuplexSuite) TestATurnThatSettlesOnOtherWordsThrowsTheGuessAway() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	guess, _, _ := listener.Eager(alice, "book a table for four")

	promoted, abandoned := listener.Settled(alice, "book a table for four on Friday")

	s.Empty(promoted, "an answer to something they did not say is worse than no answer")
	s.Equal(guess, abandoned)
}

func (s *DuplexSuite) TestCarryingOnTalkingThrowsTheGuessAway() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	guess, _, _ := listener.Eager(alice, "book a table for four")

	abandoned := listener.Began(alice)

	s.Equal(guess, abandoned, "they had not finished, so the reply was to half a sentence")
	promoted, _ := listener.Settled(alice, "book a table for four")
	s.Empty(promoted, "and it is not promoted later either")
}

func (s *DuplexSuite) TestATurnWithNoGuessSettlesQuietly() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}

	promoted, abandoned := listener.Settled(alice, "book a table")

	s.Empty(promoted)
	s.Empty(abandoned)
}

func (s *DuplexSuite) TestTwoPeopleTalkingAreTwoSeparateTurns() {
	listener := newDuplex(DuplexOptions{Speculate: true})
	alice := stt.Participant{ID: "alice"}
	bob := stt.Participant{ID: "bob"}
	guess, _, _ := listener.Eager(alice, "book a table")

	listener.Began(bob)

	promoted, _ := listener.Settled(alice, "book a table")
	s.Equal(guess, promoted, "one person talking does not cancel what someone else said")
}
