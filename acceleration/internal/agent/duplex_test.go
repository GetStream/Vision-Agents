package agent

import (
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
	listener := newDuplex(DuplexOptions{})
	alice := stt.Participant{ID: "alice"}

	s.Empty(listener.Heard(alice, "so I was wondering whether you could help me with this", true))
}

func (s *DuplexSuite) TestALongActiveGapGetsAListeningAcknowledgement() {
	listener := newDuplex(DuplexOptions{
		Backchannel:    true,
		BackchannelGap: time.Millisecond,
	})
	alice := stt.Participant{ID: "alice"}

	s.NotEmpty(listener.Presence(alice, time.Now().Add(-time.Second), true))
}

func (s *DuplexSuite) TestPresenceDoesNotTalkOverTheAgent() {
	listener := newDuplex(DuplexOptions{
		Backchannel:    true,
		BackchannelGap: time.Millisecond,
	})

	s.Empty(listener.Presence(stt.Participant{ID: "alice"}, time.Now().Add(-time.Second), false))
}
